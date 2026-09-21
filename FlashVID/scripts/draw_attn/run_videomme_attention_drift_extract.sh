#!/usr/bin/env bash
set -euo pipefail

# Usage:
# TEACHER_PRETRAINED=... STUDENT_PRETRAINED=... \
# bash FlashVID/scripts/draw_attn/run_videomme_attention_drift_extract.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER_PRETRAINED="${TEACHER_PRETRAINED:-lmms-lab/LLaVA-Video-7B-Qwen2}"
STUDENT_PRETRAINED="${STUDENT_PRETRAINED:-/path/to/merged_model}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/attn/attention_drift_200.npz}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
DEVICE="${DEVICE:-cuda:0}"  # fallback: when TEACHER/ STUDENT device not set
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:0}"
STUDENT_DEVICE="${STUDENT_DEVICE:-cuda:1}"
DTYPE="${DTYPE:-fp16}"
HF_HOME="${HF_HOME:-/path/to/huggingface/cache}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/path/to/huggingface/cache/videomme}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
VIDEO_FPS="${VIDEO_FPS:-1}"
FORCE_SAMPLE="${FORCE_SAMPLE:-true}"
MM_SPATIAL_POOL_MODE="${MM_SPATIAL_POOL_MODE:-average}"
MM_NEWLINE_POSITION="${MM_NEWLINE_POSITION:-frame}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

ENABLE_FLASHVID_STUDENT="${ENABLE_FLASHVID_STUDENT:-true}"
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
ALIGN_METHOD="${ALIGN_METHOD:-scatter}" # scatter | bilinear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

CMD=(
  python3 FlashVID/scripts/draw_attn/extract_videomme_attention_drift.py
  --teacher-pretrained "${TEACHER_PRETRAINED}"
  --student-pretrained "${STUDENT_PRETRAINED}"
  --output-path "${OUTPUT_PATH}"
  --num-samples "${NUM_SAMPLES}"
  --dataset-cache-dir "${DATASET_CACHE_DIR}"
  --hf-home "${HF_HOME}"
  --device "${DEVICE}"
  --teacher-device "${TEACHER_DEVICE}"
  --student-device "${STUDENT_DEVICE}"
  --dtype "${DTYPE}"
  --conv-template "${CONV_TEMPLATE}"
  --max-frames-num "${MAX_FRAMES_NUM}"
  --video-fps "${VIDEO_FPS}"
  --mm-spatial-pool-mode "${MM_SPATIAL_POOL_MODE}"
  --mm-newline-position "${MM_NEWLINE_POSITION}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --enable-flashvid-student "${ENABLE_FLASHVID_STUDENT}"
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
  --align-method "${ALIGN_METHOD}"
)

if [[ "${FORCE_SAMPLE}" == "true" ]]; then
  CMD+=(--force-sample)
fi

echo "[INFO] Extract VideoMME attention drift:"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
echo "[INFO] DEVICE=${DEVICE} TEACHER_DEVICE=${TEACHER_DEVICE} STUDENT_DEVICE=${STUDENT_DEVICE}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
