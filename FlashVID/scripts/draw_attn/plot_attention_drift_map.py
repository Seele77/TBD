#!/usr/bin/env python3
"""Visualize teacher/student attention drift with three-panel map.

Auto-selection mode (--auto-best):
  Scans all (sample, frame) pairs and picks the one with the smallest
  per-frame drift sum (i.e., where student attention best matches teacher).
  Teacher/student maps are normalised per-frame before comparison so that
  bright / dim frames don't dominate the ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _to_patch_map(frame_patch: np.ndarray) -> np.ndarray:
    patch_num = frame_patch.shape[0]
    side = int(np.sqrt(patch_num))
    if side * side == patch_num:
        return frame_patch.reshape(side, side)
    return frame_patch.reshape(1, patch_num)


def _load_sample_meta(npz_path: str, sample_index: int) -> dict:
    """Read per-sample metadata from .meta.json."""
    meta_path = Path(npz_path).with_suffix(".meta.json")
    if not meta_path.exists():
        print(f"[WARN] Meta file not found: {meta_path}; sample info will be omitted.")
        return {}
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    records = meta.get("records", [])
    if sample_index >= len(records):
        print(f"[WARN] sample_index={sample_index} out of range (meta has {len(records)} records).")
        return {}
    rec = records[sample_index]
    print(
        f"[INFO] Sample {sample_index}: question_id={rec.get('question_id', 'N/A')}  "
        f"video_id={rec.get('video_id', 'N/A')}  JS={rec.get('js_divergence', float('nan')):.5f}"
    )
    return rec


def _find_best_sample_frame(
    teacher_maps: np.ndarray,
    student_maps: np.ndarray,
) -> tuple[int, int]:
    """Return (sample_idx, frame_idx) with the smallest per-frame-normalised drift.

    Each frame is independently normalised to [0, 1] before computing the
    absolute difference, so that frames where both models simply have low
    activation don't get a spuriously low score.

    Require teacher_max > threshold to skip blank / near-black frames.
    Score = mean(abs(t_norm - s_norm)); lower is better.
    """
    eps = 1e-8
    blank_threshold = 1e-4   # skip frames where teacher is essentially blank
    N, F = teacher_maps.shape[:2]
    best_score = float("inf")
    best_s, best_f = 0, 0
    for s in range(N):
        for f in range(F):
            t_frame = teacher_maps[s, f]
            s_frame = student_maps[s, f]
            t_max = float(t_frame.max())
            if t_max < blank_threshold:
                continue
            t_norm = t_frame / (t_max + eps)
            s_norm = s_frame / (float(s_frame.max()) + eps)
            score = float(np.mean(np.abs(t_norm - s_norm)))
            if score < best_score:
                best_score = score
                best_s, best_f = s, f
    print(
        f"[INFO] Auto-best: sample={best_s}, frame={best_f}, "
        f"mean_normalised_drift={best_score:.5f}"
    )
    return best_s, best_f


def main(args: argparse.Namespace) -> None:
    data = np.load(args.input_npz)
    teacher_maps = data["teacher_maps"]   # [N, F, P]
    student_maps = data["student_maps"]   # [N, F, P]
    drift_maps   = data["drift_maps"]     # [N, F, P]
    js_values    = data["js_divergence"]  # [N]

    n = teacher_maps.shape[0]

    if args.auto_best:
        idx, frame_idx = _find_best_sample_frame(teacher_maps, student_maps)
    else:
        idx = min(max(args.sample_index, 0), n - 1)
        teacher = teacher_maps[idx]
        if args.frame_index < 0:
            # default: pick most salient frame for teacher
            frame_idx = int(np.argmax(teacher.sum(axis=1)))
        else:
            frame_idx = min(max(args.frame_index, 0), teacher.shape[0] - 1)

    from scipy.ndimage import gaussian_filter  # Lazy import; scipy may not be pre-loaded.

    sample_meta = _load_sample_meta(args.input_npz, idx)

    t_map = _to_patch_map(teacher_maps[idx, frame_idx])
    s_map = _to_patch_map(student_maps[idx, frame_idx])

    # Per-frame normalisation.
    eps = 1e-8
    t_norm = t_map / (np.max(t_map) + eps)
    s_norm = s_map / (np.max(s_map) + eps)
    # Visual post-processing. Gamma compression (gamma < 1) reveals faint attention without clipping peaks.
    gamma = 0.45
    t_disp = np.power(t_norm, gamma)
    s_disp = np.power(s_norm, gamma)

    # Gaussian blur: smooth blocky patch grid for a cleaner paper figure.
    sigma = 0.8
    t_disp = gaussian_filter(t_disp, sigma=sigma)
    s_disp = gaussian_filter(s_disp, sigma=sigma)

    # Plot.
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 5.0),
        dpi=150,
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes:
        ax.set_facecolor("#FAFAFA")

    _panel_kw = dict(linewidths=0, rasterized=True)

    # Teacher & Student: shared colour scale
    vmax_ts = float(max(np.max(t_disp), np.max(s_disp), eps))
    for ax, data, title in zip(
        axes,
        [t_disp, s_disp],
        ["Teacher Attention Map", "Student Attention Map"],
    ):
        sns.heatmap(
            data, ax=ax, cmap="YlOrRd",
            vmin=0.0, vmax=vmax_ts,
            cbar=True, square=True,
            cbar_kws={"shrink": 0.82, "pad": 0.02},
            **_panel_kw,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Patch column", fontsize=9, labelpad=4)
        ax.set_ylabel("Patch row", fontsize=9, labelpad=4)
        ax.tick_params(axis="both", labelsize=7)

    qid_str = sample_meta.get("question_id", "")
    vid_str = sample_meta.get("video_id", "")
    meta_suffix = f"  qid={qid_str}  vid={vid_str}" if qid_str or vid_str else ""
    sel_mode = "auto-best" if args.auto_best else "manual"

    fig.suptitle(
        f"Attention Drift Map  [{sel_mode}  sample={idx}  frame={frame_idx}"
        f"  JS={js_values[idx]:.4f}]{meta_suffix}",
        fontsize=10, y=1.01, color="#333333",
    )

    plt.tight_layout()

    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight")
    print(f"[OK] Figure saved to: {output_image}")

    report = {
        "input_npz": args.input_npz,
        "selection_mode": sel_mode,
        "sample_index": idx,
        "frame_index": frame_idx,
        "question_id": sample_meta.get("question_id", None),
        "video_id": sample_meta.get("video_id", None),
        "js_for_sample": float(js_values[idx]),
        "js_mean_all": float(np.mean(js_values)),
        "js_std_all": float(np.std(js_values)),
        "num_samples": int(n),
        "output_image": str(output_image),
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Report saved to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot attention drift three-panel map.")
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument(
        "--auto-best",
        action="store_true",
        help="Automatically pick the (sample, frame) with the smallest "
             "normalised drift across all 200 samples and frames.",
    )
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Sample index (ignored when --auto-best is set).")
    parser.add_argument("--frame-index", type=int, default=-1,
                        help="Frame index; -1 = most salient teacher frame "
                             "(ignored when --auto-best is set).")
    parser.add_argument(
        "--output-image",
        type=str,
        default="./logs/videomme_attn/attention_drift_map.png",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./logs/videomme_attn/attention_drift_map.json",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
