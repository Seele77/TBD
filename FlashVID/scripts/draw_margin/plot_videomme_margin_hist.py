#!/usr/bin/env python3
"""Plot GT margin histogram: baseline vs ours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _load_margins(path: str) -> tuple[np.ndarray, list[str]]:
    """Return (margins_array, question_id_list).

    question_ids are read from the companion .meta.json if it exists,
    otherwise an empty list is returned and alignment cannot be verified.
    """
    data = np.load(path)
    if "margins" not in data:
        raise ValueError(f"`margins` key not found in {path}")
    margins = data["margins"].astype(np.float32, copy=False)

    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        qids = [str(r["question_id"]) for r in meta.get("records", [])]
    else:
        qids = []
    return margins, qids


def _verify_alignment(
    names: list[str],
    qid_lists: list[list[str]],
    n: int,
) -> None:
    """Check that the first n question_ids are identical across all files."""
    valid = [(name, qids) for name, qids in zip(names, qid_lists) if qids]
    if len(valid) < 2:
        print("[WARN] Meta files missing for some inputs; skipping question_id alignment check.")
        return
    ref_name, ref_ids = valid[0]
    for name, qids in valid[1:]:
        if qids[:n] != ref_ids[:n]:
            mismatches = [
                (i, qids[i], ref_ids[i])
                for i in range(min(n, len(qids), len(ref_ids)))
                if qids[i] != ref_ids[i]
            ]
            raise ValueError(
                f"question_id mismatch between '{ref_name}' and '{name}'!\n"
                f"First mismatch (index, {name}, {ref_name}): {mismatches[:5]}\n"
                "Make sure both extraction runs use the same --num-samples and dataset."
            )
    print(f"[OK] question_id alignment verified for {n} samples.")


def _stats(m: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
        "negative_count": int(np.sum(m < 0)),
        "negative_ratio": float(np.mean(m < 0)),
        "sample_count": int(len(m)),
    }


def main(args: argparse.Namespace) -> None:
    baseline, baseline_qids = _load_margins(args.baseline_margin)
    ours, ours_qids = _load_margins(args.ours_margin)
    n = min(len(baseline), len(ours), args.num_points)

    _verify_alignment(
        names=["baseline", "ours"],
        qid_lists=[baseline_qids, ours_qids],
        n=n,
    )

    baseline = baseline[:n]
    ours = ours[:n]

    b_stats = _stats(baseline)
    o_stats = _stats(ours)

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=120)

    bins = args.bins
    min_x = float(min(np.min(baseline), np.min(ours)))
    max_x = float(max(np.max(baseline), np.max(ours)))
    if min_x == max_x:
        max_x = min_x + 1.0

    ax1.hist(
        baseline,
        bins=bins,
        range=(min_x, max_x),
        alpha=0.45,
        color="#e74c3c",
        label="Baseline",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.hist(
        ours,
        bins=bins,
        range=(min_x, max_x),
        alpha=0.45,
        color="#2ecc71",
        label="Ours",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    sns.kdeplot(baseline, ax=ax1, color="#c0392b", lw=2)
    sns.kdeplot(ours, ax=ax1, color="#27ae60", lw=2)
    ax1.axvline(0.0, color="black", linestyle="--", lw=1.4, alpha=0.8, label="Decision boundary (margin=0)")
    ax1.axvline(b_stats["mean"], color="#c0392b", linestyle=":", lw=2)
    ax1.axvline(o_stats["mean"], color="#27ae60", linestyle=":", lw=2)
    ax1.set_title("GT-anchored Margin Distribution", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel(r"$\Delta = L_{gt} - L_{max\_wrong}$")
    ax1.set_ylabel("Density")
    ax1.legend()

    categories = ["Mean Margin", "Negative Ratio (delta<0)"]
    baseline_vals = [b_stats["mean"], b_stats["negative_ratio"]]
    ours_vals = [o_stats["mean"], o_stats["negative_ratio"]]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width / 2, baseline_vals, width, color="#e74c3c", alpha=0.8, label="Baseline")
    ax2.bar(x + width / 2, ours_vals, width, color="#2ecc71", alpha=0.8, label="Ours")
    for i, v in enumerate(baseline_vals):
        ax2.text(i - width / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    for i, v in enumerate(ours_vals):
        ax2.text(i + width / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_title("Margin Summary", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Value")
    ax2.legend()

    plt.tight_layout()

    out_img = Path(args.output_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, bbox_inches="tight")
    print(f"[OK] Figure saved to: {out_img}")

    report = {
        "num_points": n,
        "baseline": b_stats,
        "ours": o_stats,
        "improvement": {
            "mean_margin_delta": float(o_stats["mean"] - b_stats["mean"]),
            "negative_ratio_drop": float(b_stats["negative_ratio"] - o_stats["negative_ratio"]),
        },
        "baseline_margin_file": args.baseline_margin,
        "ours_margin_file": args.ours_margin,
        "output_image": str(out_img),
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Report saved to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot baseline vs ours margin histogram.")
    parser.add_argument("--baseline-margin", type=str, required=True)
    parser.add_argument("--ours-margin", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument(
        "--output-image",
        type=str,
        default="./logs/videomme_margin/margin_hist_baseline_vs_ours.png",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./logs/videomme_margin/margin_hist_baseline_vs_ours.json",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
