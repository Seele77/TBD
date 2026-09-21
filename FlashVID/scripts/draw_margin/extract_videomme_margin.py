#!/usr/bin/env python3
"""Extract GT-anchored margin statistics on VideoMME."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
LMMS_EVAL_ROOT = REPO_ROOT / "FlashVID" / "lmms-eval"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from lmms_eval.models.simple.llava_vid import LlavaVid  # noqa: E402
from lmms_eval.tasks.videomme.utils import videomme_doc_to_text  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def _build_prompt(model: LlavaVid, question_text: str) -> str:
    qs = question_text
    if model.model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[model.conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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


def _token_id_for_choice(tokenizer: Any, letter: str) -> int:
    candidates = [letter, f" {letter}", f"\n{letter}"]
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return ids[-1]
    raise ValueError(f"Failed to map choice token: {letter}")


def extract_margin(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    model = LlavaVid(
        pretrained=args.pretrained,
        conv_template=args.conv_template,
        max_frames_num=args.max_frames_num,
        video_fps=args.video_fps,
        force_sample=args.force_sample,
        add_time_instruction=args.add_time_instruction,
        mm_spatial_pool_mode=args.mm_spatial_pool_mode,
        mm_newline_position=args.mm_newline_position,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        device=args.device,
        device_map=args.device,
        enable_flashvid=args.enable_flashvid,
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

    print("[INFO] Model loaded. Start loading dataset...")
    cache_dir = os.path.expanduser(args.dataset_cache_dir)
    hf_home = os.path.expanduser(args.hf_home)
    docs = _load_docs_from_parquet(hf_home=hf_home, num_samples=args.num_samples)
    if len(docs) < args.num_samples:
        raise ValueError(
            f"Requested num_samples={args.num_samples}, but only got {len(docs)} from parquet"
        )
    print(f"[INFO] Dataset ready. Collected docs: {len(docs)}")

    choice_ids = {
        letter: _token_id_for_choice(model.tokenizer, letter)
        for letter in ("A", "B", "C", "D")
    }

    margins: list[float] = []
    records: list[dict[str, Any]] = []

    for idx in tqdm(range(args.num_samples), desc="Extracting margins"):
        doc = docs[idx]
        gt = str(doc["answer"]).strip().upper()
        if gt not in choice_ids:
            continue

        video_path = _resolve_video_path(cache_dir, doc["videoID"])
        question_text = videomme_doc_to_text(
            doc, lmms_eval_specific_kwargs={"post_prompt": "The best answer is:"}
        )
        prompt = _build_prompt(model, question_text)
        input_ids = (
            tokenizer_image_token(
                prompt,
                model.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(model.device)
        )

        video, _, _ = model.load_video(
            video_path,
            model.max_frames_num,
            model.fps,
            force_sample=model.force_sample,
        )
        video_tensor = model._image_processor.preprocess(
            video, return_tensors="pt"
        )["pixel_values"].to(model.device)
        video_tensor = (
            video_tensor.bfloat16()
            if model.torch_dtype == "bfloat16"
            else video_tensor.half()
        )
        videos = [video_tensor]

        with torch.inference_mode():
            outputs = model.model(
                input_ids=input_ids,
                images=videos,
                modalities="video",
            )
        logits = outputs["logits"][0, -1, :].float().detach().cpu()

        gt_logit = float(logits[choice_ids[gt]].item())
        wrong_letters = [x for x in ("A", "B", "C", "D") if x != gt]
        wrong_logits = [float(logits[choice_ids[x]].item()) for x in wrong_letters]
        max_wrong_logit = max(wrong_logits)
        hard_negative = wrong_letters[int(np.argmax(wrong_logits))]
        margin = gt_logit - max_wrong_logit

        pred = max(
            ("A", "B", "C", "D"),
            key=lambda x: float(logits[choice_ids[x]].item()),
        )

        margins.append(margin)
        records.append(
            {
                "index": idx,
                "question_id": doc["question_id"],
                "video_id": doc["videoID"],
                "gt": gt,
                "pred": pred,
                "correct": pred == gt,
                "gt_logit": gt_logit,
                "max_wrong_logit": max_wrong_logit,
                "hard_negative": hard_negative,
                "margin": margin,
            }
        )

    margin_arr = np.asarray(margins, dtype=np.float32)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        margins=margin_arr,
        sample_count=np.array([len(margin_arr)], dtype=np.int32),
        model_path=np.array([args.pretrained]),
        margin_mean=np.array([float(np.mean(margin_arr))], dtype=np.float32),
        margin_std=np.array([float(np.std(margin_arr))], dtype=np.float32),
        negative_ratio=np.array([float(np.mean(margin_arr < 0))], dtype=np.float32),
    )

    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(margin_arr),
                "model_path": args.pretrained,
                "output_npz": str(output_path),
                "choice_token_ids": choice_ids,
                "margin_mean": float(np.mean(margin_arr)),
                "margin_std": float(np.std(margin_arr)),
                "negative_count": int(np.sum(margin_arr < 0)),
                "negative_ratio": float(np.mean(margin_arr < 0)),
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved margins to: {output_path}")
    print(f"[OK] Saved metadata to: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GT-anchored margin for VideoMME."
    )
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--dataset-cache-dir", type=str, default="videomme",
                        help="Path to VideoMME video cache (contains data/*.mp4)")
    parser.add_argument("--hf-home", type=str,
                        default=os.environ.get("HF_HOME", "~/.cache/huggingface"),
                        help="HuggingFace home dir (hub parquet files are read from here)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--max-frames-num", type=int, default=64)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--add-time-instruction", action="store_true")
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="frame")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")

    parser.add_argument("--enable-flashvid", action="store_true")
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
    return parser


if __name__ == "__main__":
    extract_margin(build_parser().parse_args())
