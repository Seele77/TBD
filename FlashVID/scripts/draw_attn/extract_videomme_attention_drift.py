#!/usr/bin/env python3
"""Extract Teacher/Student attention drift statistics on VideoMME."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
LMMS_EVAL_ROOT = REPO_ROOT / "FlashVID" / "lmms-eval"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from flashvid.utils import flashvid_compression  # noqa: E402
from lmms_eval.models.simple.llava_vid import LlavaVid  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def _load_docs_from_parquet(hf_home: str, num_samples: int) -> list[dict]:
    """Read VideoMME test metadata directly from HF hub parquet files."""
    hub_root = Path(hf_home) / "hub" / "datasets--lmms-lab--Video-MME"
    if not hub_root.exists():
        raise FileNotFoundError(f"HF hub cache not found: {hub_root}")

    parquet_files: list[Path] = []
    for pattern in (
        "snapshots/*/data/test-*.parquet",
        "snapshots/*/*/test-*.parquet",
        "snapshots/*/test-*.parquet",
        "snapshots/*/*/*.parquet",
        "snapshots/*/*.parquet",
    ):
        found = sorted(hub_root.glob(pattern))
        if found:
            parquet_files = found
            break

    if not parquet_files:
        all_parquets = sorted(hub_root.rglob("*.parquet"))
        parquet_files = [p for p in all_parquets if "test" in p.name.lower()] or all_parquets

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under: {hub_root}\n"
            "Contents:\n" + "\n".join(str(p) for p in sorted(hub_root.rglob("*"))[:30])
        )

    print(f"[INFO] Reading from parquet: {[p.name for p in parquet_files]}")
    import pandas as pd  # lazy import: must come after torch to avoid static TLS exhaustion
    docs: list[dict] = []
    for pf in sorted(parquet_files):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            docs.append(row.to_dict())
            if len(docs) >= num_samples:
                return docs
    return docs


def _resolve_video_path(cache_dir: str, video_id: str) -> str:
    data_root = Path(cache_dir) / "data"
    for ext in ("mp4", "MP4", "mkv"):
        p = data_root / f"{video_id}.{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Video file not found under {data_root} for video_id={video_id}"
    )


def _normalize_probs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    s = torch.sum(x)
    if float(s.item()) <= 0.0:
        x = torch.ones_like(x) / x.numel()
    else:
        x = x / s
    return torch.clamp(x, min=eps)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = _normalize_probs(p, eps=eps)
    q = _normalize_probs(q, eps=eps)
    m = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * torch.log(p / m)) + 0.5 * torch.sum(q * torch.log(q / m))
    return float(js.item())


def _build_model(
    pretrained: str,
    device: str,
    dtype: str,
    conv_template: str,
    max_frames_num: int,
    video_fps: int,
    force_sample: bool,
    mm_spatial_pool_mode: str,
    mm_newline_position: str,
    attn_implementation: str,
    enable_flashvid: bool,
    retention_ratio: float,
    do_segment: bool,
    min_segment_num: int,
    complementary_segment: bool,
    token_selection_method: str,
    alpha: float,
    temporal_threshold: float,
    expansion: float,
    pruning_layer: int,
    llm_retention_ratio: float,
) -> LlavaVid:
    torch_dtype = "bfloat16" if dtype == "bf16" else "float16"
    model = LlavaVid(
        pretrained=pretrained,
        conv_template=conv_template,
        max_frames_num=max_frames_num,
        video_fps=video_fps,
        force_sample=force_sample,
        add_time_instruction=False,
        mm_spatial_pool_mode=mm_spatial_pool_mode,
        mm_newline_position=mm_newline_position,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device,
        enable_flashvid=enable_flashvid,
        retention_ratio=retention_ratio,
        do_segment=do_segment,
        min_segment_num=min_segment_num,
        complementary_segment=complementary_segment,
        token_selection_method=token_selection_method,
        alpha=alpha,
        temporal_threshold=temporal_threshold,
        expansion=expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
    )
    return model


def _pooled_cls_map(model: LlavaVid, video_tensor: torch.Tensor) -> torch.Tensor:
    vision_tower = model.model.get_model().get_vision_tower()
    _, cls_attn = vision_tower(video_tensor)
    pooled_cls = model.model.get_2dPool(cls_attn.unsqueeze(-1)).squeeze(-1)
    return pooled_cls.float()


def _recover_student_attention(
    teacher_map: torch.Tensor,
    student_cls_map: torch.Tensor,
    keep_visual_indices: torch.Tensor,
    align_method: str,
) -> torch.Tensor:
    num_frames, num_visual_tokens = teacher_map.shape
    full_len = num_frames * num_visual_tokens

    student_flat_full = student_cls_map.reshape(-1)
    kept_scores = student_flat_full[keep_visual_indices]

    if align_method == "scatter":
        recovered = torch.zeros(full_len, dtype=kept_scores.dtype, device=kept_scores.device)
        recovered[keep_visual_indices] = kept_scores
        return recovered.reshape(num_frames, num_visual_tokens)

    recovered_1d = F.interpolate(
        kept_scores.view(1, 1, -1),
        size=full_len,
        mode="linear",
        align_corners=False,
    ).view(full_len)
    return recovered_1d.reshape(num_frames, num_visual_tokens)


def extract_attention_drift(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    teacher_device = args.teacher_device or args.device
    student_device = args.student_device or args.device
    print(f"[INFO] Device placement: teacher={teacher_device}, student={student_device}")

    teacher = _build_model(
        pretrained=args.teacher_pretrained,
        device=teacher_device,
        dtype=args.dtype,
        conv_template=args.conv_template,
        max_frames_num=args.max_frames_num,
        video_fps=args.video_fps,
        force_sample=args.force_sample,
        mm_spatial_pool_mode=args.mm_spatial_pool_mode,
        mm_newline_position=args.mm_newline_position,
        attn_implementation=args.attn_implementation,
        enable_flashvid=False,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        expansion=args.expansion,
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
    )
    student = _build_model(
        pretrained=args.student_pretrained,
        device=student_device,
        dtype=args.dtype,
        conv_template=args.conv_template,
        max_frames_num=args.max_frames_num,
        video_fps=args.video_fps,
        force_sample=args.force_sample,
        mm_spatial_pool_mode=args.mm_spatial_pool_mode,
        mm_newline_position=args.mm_newline_position,
        attn_implementation=args.attn_implementation,
        enable_flashvid=args.enable_flashvid_student,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        expansion=args.expansion,
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
    )

    print("[INFO] Models loaded. Start loading dataset...")
    cache_dir = os.path.expanduser(args.dataset_cache_dir)
    hf_home = os.path.expanduser(args.hf_home)
    docs = _load_docs_from_parquet(hf_home=hf_home, num_samples=args.num_samples)
    if len(docs) < args.num_samples:
        raise ValueError(
            f"Requested num_samples={args.num_samples}, but only got {len(docs)} from parquet"
        )
    print(f"[INFO] Dataset ready. Collected docs: {len(docs)}")
    n = len(docs)

    teacher_maps: list[np.ndarray] = []
    student_maps: list[np.ndarray] = []
    drift_maps: list[np.ndarray] = []
    js_values: list[float] = []
    records: list[dict[str, Any]] = []

    for idx in tqdm(range(n), desc="Extracting attention drift"):
        doc = docs[idx]
        video_path = _resolve_video_path(cache_dir, doc["videoID"])
        video, _, _ = teacher.load_video(
            video_path,
            teacher.max_frames_num,
            teacher.fps,
            force_sample=teacher.force_sample,
        )

        teacher_video = teacher._image_processor.preprocess(video, return_tensors="pt")[
            "pixel_values"
        ].to(teacher.device)
        teacher_video = (
            teacher_video.bfloat16()
            if teacher.torch_dtype == "bfloat16"
            else teacher_video.half()
        )

        student_video = student._image_processor.preprocess(video, return_tensors="pt")[
            "pixel_values"
        ].to(student.device)
        student_video = (
            student_video.bfloat16()
            if student.torch_dtype == "bfloat16"
            else student_video.half()
        )

        with torch.inference_mode():
            teacher_map = _pooled_cls_map(teacher, teacher_video)  # [F, P]
            student_map_full = _pooled_cls_map(student, student_video)  # [F, P]

            student_model_core = student.model.get_model()
            student_image_features, student_cls = student_model_core.get_vision_tower()(
                student_video
            )
            student_image_features = student_model_core.mm_projector(student_image_features)
            pooled_feat = student.model.get_2dPool(student_image_features)
            pooled_cls = student.model.get_2dPool(student_cls.unsqueeze(-1)).squeeze(-1)

            student.model.flashvid_config.return_compression_metadata = True
            compression_out = flashvid_compression(
                video_features=pooled_feat,
                cls_attention=pooled_cls,
                flashvid_config=student.model.flashvid_config,
                return_metadata=True,
            )
            _, keep_visual_indices, meta = compression_out

            recovered_student_map = _recover_student_attention(
                teacher_map=teacher_map,
                student_cls_map=student_map_full,
                keep_visual_indices=keep_visual_indices.long(),
                align_method=args.align_method,
            )

            t_prob = _normalize_probs(teacher_map.reshape(-1))
            s_prob = _normalize_probs(recovered_student_map.reshape(-1))
            js = _js_divergence(t_prob, s_prob)

        drift = torch.abs(t_prob - s_prob).reshape_as(teacher_map)
        teacher_maps.append(t_prob.reshape_as(teacher_map).cpu().numpy())
        student_maps.append(s_prob.reshape_as(teacher_map).cpu().numpy())
        drift_maps.append(drift.cpu().numpy())
        js_values.append(js)
        records.append(
            {
                "index": idx,
                "question_id": doc["question_id"],
                "video_id": doc["videoID"],
                "js_divergence": js,
                "num_frames": int(meta["num_frames"]),
                "num_visual_tokens": int(meta["num_visual_tokens"]),
                "retained_token_count": int(meta["retained_token_count"]),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        teacher_maps=np.stack(teacher_maps).astype(np.float32, copy=False),
        student_maps=np.stack(student_maps).astype(np.float32, copy=False),
        drift_maps=np.stack(drift_maps).astype(np.float32, copy=False),
        js_divergence=np.asarray(js_values, dtype=np.float32),
        sample_count=np.array([n], dtype=np.int32),
        js_mean=np.array([float(np.mean(js_values))], dtype=np.float32),
        js_std=np.array([float(np.std(js_values))], dtype=np.float32),
        align_method=np.array([args.align_method]),
    )

    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher_model": args.teacher_pretrained,
                "student_model": args.student_pretrained,
                "num_samples": n,
                "align_method": args.align_method,
                "js_mean": float(np.mean(js_values)),
                "js_std": float(np.std(js_values)),
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved attention drift data to: {output_path}")
    print(f"[OK] Saved metadata to: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract attention drift maps and JS divergence on VideoMME."
    )
    parser.add_argument("--teacher-pretrained", type=str, required=True)
    parser.add_argument("--student-pretrained", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--dataset-cache-dir", type=str, default="videomme",
                        help="Path to VideoMME video cache (contains data/*.mp4)")
    parser.add_argument("--hf-home", type=str,
                        default=os.environ.get("HF_HOME", "~/.cache/huggingface"),
                        help="HuggingFace home dir (hub parquet files are read from here)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher-device", type=str, default="",
                        help="Dedicated device for teacher model, e.g. cuda:0")
    parser.add_argument("--student-device", type=str, default="",
                        help="Dedicated device for student model, e.g. cuda:1")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--max-frames-num", type=int, default=64)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="frame")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")

    parser.add_argument("--enable-flashvid-student", type=_str2bool, default=True)
    parser.add_argument("--retention-ratio", type=float, default=0.1)
    parser.add_argument("--do-segment", type=_str2bool, default=True)
    parser.add_argument("--complementary-segment", type=_str2bool, default=True)
    parser.add_argument("--min-segment-num", type=int, default=8)
    parser.add_argument("--token-selection-method", type=str, default="attn_div_v2")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temporal-threshold", type=float, default=0.8)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.3)
    parser.add_argument(
        "--align-method",
        type=str,
        default="scatter",
        choices=["scatter", "bilinear"],
    )
    return parser


if __name__ == "__main__":
    extract_attention_drift(build_parser().parse_args())
