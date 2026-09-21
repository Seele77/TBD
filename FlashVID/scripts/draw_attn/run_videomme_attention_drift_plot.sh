#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   # Recommended: auto-select the best (sample, frame) pair
#   INPUT_NPZ=./logs/attn/attention_drift_200.npz \
#   bash FlashVID/scripts/draw_attn/run_videomme_attention_drift_plot.sh
#
#   # Manual: specify sample and frame
#   AUTO_BEST=false SAMPLE_INDEX=42 FRAME_INDEX=3 \
#   INPUT_NPZ=./logs/attn/attention_drift_200.npz \
#   bash FlashVID/scripts/draw_attn/run_videomme_attention_drift_plot.sh

INPUT_NPZ="${INPUT_NPZ:-./logs/attn/attention_drift_200.npz}"
AUTO_BEST="${AUTO_BEST:-true}"     # true = scan all samples and frames for smallest drift
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"  # used only when AUTO_BEST=false
FRAME_INDEX="${FRAME_INDEX:--1}"   # -1 = most salient teacher frame; used when AUTO_BEST=false
OUTPUT_IMAGE="${OUTPUT_IMAGE:-./logs/attn/attention_drift_map.png}"
OUTPUT_REPORT="${OUTPUT_REPORT:-./logs/attn/attention_drift_map.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CMD=(
  python3 FlashVID/scripts/draw_attn/plot_attention_drift_map.py
  --input-npz "${INPUT_NPZ}"
  --sample-index "${SAMPLE_INDEX}"
  --frame-index "${FRAME_INDEX}"
  --output-image "${OUTPUT_IMAGE}"
  --output-report "${OUTPUT_REPORT}"
)

if [[ "${AUTO_BEST}" == "true" ]]; then
  CMD+=(--auto-best)
fi

echo "[INFO] Plotting attention drift map:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
