#!/usr/bin/env bash
# Merge LoRA adapter (from run_llava_video_flashvid_lora_kd.sh training) with base model.
# Uses same MODEL_NAME and OUTPUT_DIR as the training script by default.
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-lmms-lab/LLaVA-Video-7B-Qwen2}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/anonymous_experiment_video}
MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR:-${OUTPUT_DIR}_merged}

echo "Base model:      ${MODEL_NAME}"
echo "Adapter path:   ${OUTPUT_DIR}"
echo "Merged output:  ${MERGED_OUTPUT_DIR}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

python "${SCRIPT_DIR}/merge_lora.py" \
  --base_model "${MODEL_NAME}" \
  --adapter_path "${OUTPUT_DIR}" \
  --output_dir "${MERGED_OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --bf16

echo ""
echo "Merged model saved to: ${MERGED_OUTPUT_DIR}"
