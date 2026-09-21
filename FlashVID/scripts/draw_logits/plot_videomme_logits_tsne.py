#!/usr/bin/env python3
"""Plot teacher/student logits distribution alignment on VideoMME.

Distance metric: KL divergence restricted to A/B/C/D choice tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_logits(path: str) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Return (logits_array, question_id_list, choice_token_ids).

    choice_token_ids maps "A"/"B"/"C"/"D" -> vocab index, read from .meta.json.
    question_ids and choice_token_ids are empty / {} when meta file is absent.
    """
    data = np.load(path)
    if "logits" not in data:
        raise ValueError(f"`logits` key not found in {path}")
    logits = data["logits"]

    meta_path = Path(path).with_suffix(".meta.json")
    qids: list[str] = []
    choice_ids: dict[str, int] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        qids = [str(r["question_id"]) for r in meta.get("records", [])]
        choice_ids = {k: int(v) for k, v in meta.get("choice_token_ids", {}).items()}
    return logits, qids, choice_ids


def _abcd_probs(logits: np.ndarray, choice_ids: dict[str, int]) -> np.ndarray:
    """Extract A/B/C/D logits and apply softmax, returning shape [N, 4]."""
    cols = [choice_ids[k] for k in ("A", "B", "C", "D")]
    raw = logits[:, cols].astype(np.float32)          # [N, 4]
    raw = raw - raw.max(axis=1, keepdims=True)         # numerical stability
    exp = np.exp(raw)
    return exp / exp.sum(axis=1, keepdims=True)        # [N, 4]


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample KL(p || q), shape [N]."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q), axis=1)


def _verify_alignment(
    names: list[str],
    qid_lists: list[list[str]],
    n: int,
) -> None:
    """Check that the first n question_ids are identical across all files."""
    valid = [qids for qids in qid_lists if qids]
    if len(valid) < 2:
        print("[WARN] Meta files missing for some inputs; skipping question_id alignment check.")
        return
    ref_ids = valid[0][:n]
    ref_name = names[qid_lists.index(valid[0])]
    for name, qids in zip(names, qid_lists):
        if not qids:
            print(f"[WARN] No meta file for {name}; cannot verify alignment.")
            continue
        if qids[:n] != ref_ids:
            mismatches = [
                (i, qids[i], ref_ids[i])
                for i in range(min(n, len(qids), len(ref_ids)))
                if qids[i] != ref_ids[i]
            ]
            raise ValueError(
                f"question_id mismatch between '{ref_name}' and '{name}'!\n"
                f"First mismatch (index, {name}, {ref_name}): {mismatches[:5]}\n"
                "Make sure all three extraction runs use the same --num-samples and dataset."
            )
    print(f"[OK] question_id alignment verified for {n} samples.")


def main(args: argparse.Namespace) -> None:
    teacher_logits, teacher_qids, teacher_cids = load_logits(args.teacher_logits)
    naive_logits,   naive_qids,   naive_cids   = load_logits(args.naive_logits)
    distilled_logits, distilled_qids, distilled_cids = load_logits(args.distilled_logits)

    n = min(len(teacher_logits), len(naive_logits), len(distilled_logits), args.num_points)

    _verify_alignment(
        names=["teacher", "naive", "distilled"],
        qid_lists=[teacher_qids, naive_qids, distilled_qids],
        n=n,
    )

    teacher_logits   = teacher_logits[:n]
    naive_logits     = naive_logits[:n]
    distilled_logits = distilled_logits[:n]

    # choice_token_ids: prefer teacher's, fall back to naive/distilled
    choice_ids = teacher_cids or naive_cids or distilled_cids
    if not choice_ids:
        raise ValueError(
            "No choice_token_ids found in any .meta.json. "
            "Re-run extraction to regenerate meta files."
        )
    print(f"[INFO] Using choice_token_ids: {choice_ids}")

    # A/B/C/D softmax probabilities [N, 4]
    t_p = _abcd_probs(teacher_logits,   choice_ids)
    n_p = _abcd_probs(naive_logits,     choice_ids)
    d_p = _abcd_probs(distilled_logits, choice_ids)

    # Per-sample KL(teacher || student) on A/B/C/D
    kl_naive      = _kl_div(t_p, n_p)   # [N]
    kl_distilled  = _kl_div(t_p, d_p)   # [N]

    win_mask = kl_distilled < kl_naive   # samples where distilled beats naive

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=120)

    # Left: per-sample scatter KL(T||N) vs KL(T||D).
    # KL divergence is heavily right-skewed; use 95th pct to zoom into main cluster.
    # A 2-D KDE density layer reveals concentration near origin that scatter alone hides.
    all_kl = np.concatenate([kl_naive, kl_distilled])
    p95 = float(np.percentile(all_kl, 95))
    ax_max = p95 * 1.10
    outlier_mask = (kl_naive > ax_max) | (kl_distilled > ax_max)
    n_outliers = int(outlier_mask.sum())
    in_range = ~outlier_mask

    # 2-D KDE density background (shows how densely packed the near-origin region is)
    try:
        sns.kdeplot(
            x=kl_naive[in_range], y=kl_distilled[in_range], ax=ax1,
            fill=True, cmap="Blues", alpha=0.35, thresh=0.05, levels=8,
            bw_adjust=0.8, zorder=1,
        )
    except Exception:
        pass  # skip if not enough unique points

    diag = np.linspace(0, ax_max, 200)
    ax1.plot(diag, diag, color="gray", linestyle="--", lw=1.5, alpha=0.7,
             label="y = x  (equal KL)", zorder=2)
    ax1.fill_between(diag, 0, diag, alpha=0.06, color="#2ecc71", zorder=0)

    # Scatter points (including outliers): clip to axis boundary for display only.
    # Win/loss counts are always computed on original KL values (no clipping).
    x_plot = np.clip(kl_naive, 0, ax_max * 0.999)
    y_plot = np.clip(kl_distilled, 0, ax_max * 0.999)
    ax1.scatter(
        x_plot[~win_mask], y_plot[~win_mask],
        c="#e74c3c", alpha=0.65, s=50, edgecolors="none",
        label=f"Naive wins ({int((~win_mask).sum())})", zorder=4,
    )
    ax1.scatter(
        x_plot[win_mask], y_plot[win_mask],
        c="#27ae60", alpha=0.85, s=70, edgecolors="white", linewidth=0.6,
        marker="*", label=f"Distilled wins ({int(win_mask.sum())})", zorder=5,
    )

    ax1.set_xlim(0, ax_max)
    ax1.set_ylim(0, ax_max)
    ax1.set_aspect("equal")
    ax1.set_title("Per-sample KL Divergence to Teacher\n(A/B/C/D choices only)",
                  fontsize=14, fontweight="bold", pad=13)
    ax1.set_xlabel(r"$\mathrm{KL}(P_T \| P_\mathrm{naive})$", fontsize=12)
    ax1.set_ylabel(r"$\mathrm{KL}(P_T \| P_\mathrm{distilled})$", fontsize=12)
    ax1.legend(loc="upper left", frameon=True, shadow=True)
    if n_outliers > 0:
        print(
            f"[INFO] Left scatter axis clipped at 95th pct ({p95:.5f}); "
            f"{n_outliers} outlier(s) are drawn on plot boundaries."
        )

    # Right: KDE of KL distributions.
    # Align x range with left plot's clipped axis so the two plots are consistent
    kl_max = ax_max
    sns.kdeplot(kl_naive, ax=ax2, fill=True, color="#e74c3c",
                label="Naive", bw_adjust=0.7, alpha=0.35, clip=(0, kl_max))
    sns.kdeplot(kl_distilled, ax=ax2, fill=True, color="#2ecc71",
                label="Distilled", bw_adjust=0.7, alpha=0.55, clip=(0, kl_max))

    mean_n = float(np.mean(kl_naive))
    mean_d = float(np.mean(kl_distilled))
    ax2.axvline(mean_n, color="#c0392b", linestyle="--", lw=2, alpha=0.85)
    ax2.axvline(mean_d, color="#27ae60",  linestyle="--", lw=2, alpha=0.85)

    y_top = ax2.get_ylim()[1]
    ax2.annotate(
        f"Mean: {mean_n:.4f}",
        xy=(mean_n, y_top * 0.78),
        xytext=(mean_n + kl_max * 0.04, y_top * 0.80),
        arrowprops=dict(arrowstyle="->", color="#c0392b"),
        color="#c0392b", fontweight="bold",
    )
    ax2.annotate(
        f"Mean: {mean_d:.4f}",
        xy=(mean_d, y_top * 0.55),
        xytext=(mean_d + kl_max * 0.04, y_top * 0.57),
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        color="#27ae60", fontweight="bold",
    )

    ax2.set_xlim(0, kl_max)
    ax2.set_title("KL Divergence Distribution to Teacher\n(A/B/C/D choices only)",
                  fontsize=14, fontweight="bold", pad=13)
    ax2.set_xlabel(r"$\mathrm{KL}(P_\mathrm{teacher} \| P_\mathrm{student})$")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    if args.output_image:
        plt.savefig(args.output_image, bbox_inches="tight")
        print(f"[OK] Figure saved to: {args.output_image}")
    else:
        plt.show()

    print(f"[INFO] Mean KL  naive={mean_n:.5f}  distilled={mean_d:.5f}  "
          f"reduction={mean_n - mean_d:.5f} ({(mean_n - mean_d) / mean_n * 100:.1f}%)")
    print(f"[INFO] Distilled wins on {win_mask.sum()}/{n} samples "
          f"({win_mask.mean() * 100:.1f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot VideoMME logits alignment figure.")
    parser.add_argument("--teacher-logits",   type=str, required=True)
    parser.add_argument("--naive-logits",     type=str, required=True)
    parser.add_argument("--distilled-logits", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-image", type=str, default="")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
