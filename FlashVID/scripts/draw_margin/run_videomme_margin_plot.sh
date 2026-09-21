#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BASELINE_MARGIN=... OURS_MARGIN=... bash FlashVID/scripts/draw_margin/run_videomme_margin_plot.sh

BASELINE_MARGIN="${BASELINE_MARGIN:-./logs/videomme_margin/baseline_margin_200.npz}"
OURS_MARGIN="${OURS_MARGIN:-./logs/videomme_margin/ours_margin_200.npz}"
NUM_POINTS="${NUM_POINTS:-200}"
BINS="${BINS:-36}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-./logs/videomme_margin/margin_hist_baseline_vs_ours.png}"
OUTPUT_REPORT="${OUTPUT_REPORT:-./logs/videomme_margin/margin_hist_baseline_vs_ours.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

python3 FlashVID/scripts/draw_margin/plot_videomme_margin_hist.py \
  --baseline-margin "${BASELINE_MARGIN}" \
  --ours-margin "${OURS_MARGIN}" \
  --num-points "${NUM_POINTS}" \
  --bins "${BINS}" \
  --output-image "${OUTPUT_IMAGE}" \
  --output-report "${OUTPUT_REPORT}"
