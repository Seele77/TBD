#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash FlashVID/scripts/draw_margin/run_videomme_margin_extract.sh
# Override by env vars.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED="${PRETRAINED:-/path/to/merged_model}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/videomme_margin/distilled_margin_200.npz}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-fp16}"
HF_HOME="${HF_HOME:-/path/to/huggingface/cache}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/path/to/huggingface/cache/videomme}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-64}"
VIDEO_FPS="${VIDEO_FPS:-1}"
FORCE_SAMPLE="${FORCE_SAMPLE:-true}"
ADD_TIME_INSTRUCTION="${ADD_TIME_INSTRUCTION:-false}"
MM_SPATIAL_POOL_MODE="${MM_SPATIAL_POOL_MODE:-average}"
MM_NEWLINE_POSITION="${MM_NEWLINE_POSITION:-frame}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

ENABLE_FLASHVID="${ENABLE_FLASHVID:-true}"
RETENTION_RATIO="${RETENTION_RATIO:-0.10}"
DO_SEGMENT="${DO_SEGMENT:-true}"
COMPLEMENTARY_SEGMENT="${COMPLEMENTARY_SEGMENT:-true}"
MIN_SEGMENT_NUM="${MIN_SEGMENT_NUM:-8}"
TOKEN_SELECTION_METHOD="${TOKEN_SELECTION_METHOD:-attn_div_v2}"
ALPHA="${ALPHA:-0.7}"
TEMPORAL_THRESHOLD="${TEMPORAL_THRESHOLD:-0.8}"
EXPANSION="${EXPANSION:-1.25}"
PRUNING_LAYER="${PRUNING_LAYER:-20}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-0.3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

CMD=(
  python3 FlashVID/scripts/draw_margin/extract_videomme_margin.py
  --pretrained "${PRETRAINED}"
  --output-path "${OUTPUT_PATH}"
  --num-samples "${NUM_SAMPLES}"
  --dataset-cache-dir "${DATASET_CACHE_DIR}"
  --hf-home "${HF_HOME}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --conv-template "${CONV_TEMPLATE}"
  --max-frames-num "${MAX_FRAMES_NUM}"
  --video-fps "${VIDEO_FPS}"
  --mm-spatial-pool-mode "${MM_SPATIAL_POOL_MODE}"
  --mm-newline-position "${MM_NEWLINE_POSITION}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --retention-ratio "${RETENTION_RATIO}"
  --do-segment "${DO_SEGMENT}"
  --complementary-segment "${COMPLEMENTARY_SEGMENT}"
  --min-segment-num "${MIN_SEGMENT_NUM}"
  --token-selection-method "${TOKEN_SELECTION_METHOD}"
  --alpha "${ALPHA}"
  --temporal-threshold "${TEMPORAL_THRESHOLD}"
  --expansion "${EXPANSION}"
  --pruning-layer "${PRUNING_LAYER}"
  --llm-retention-ratio "${LLM_RETENTION_RATIO}"
)

if [[ "${FORCE_SAMPLE}" == "true" ]]; then
  CMD+=(--force-sample)
fi
if [[ "${ADD_TIME_INSTRUCTION}" == "true" ]]; then
  CMD+=(--add-time-instruction)
fi
if [[ "${ENABLE_FLASHVID}" == "true" ]]; then
  CMD+=(--enable-flashvid)
fi

echo "[INFO] Extract VideoMME margin:"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
