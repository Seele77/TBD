#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Evaluation benchmarks.
TASKS=( ${TASKS:-videomme mvbench longvideobench_val_v egoschema} )

# Pretrained model path.
PRETRAINED="${PRETRAINED:-./outputs/tbd_onevision_r10_merged}"

# ! FlashVid arguments.
RETENTION_RATIOS=( ${FLASHVID_RETENTION_RATIOS:-0.10} )
## Dyseg (fixed)
DO_SEGMENT=True
MIN_SEGMENT_NUM=8
COMPLEMENTARY_SEGMENT=True
## ADTS and TSTM (fixed)
TOKEN_SELECTION_METHOD=attn_div_v2
TEMPORAL_THRESHOLD=0.8
ALPHA=0.7
## Inner-LLM Pruning (fixed)
EXPANSION=1.25
PRUNING_LAYER=20
LLM_RETENTION_RATIO=0.3

BASE_FLASHVID_ARGS="enable_flashvid=True,expansion=$EXPANSION,do_segment=$DO_SEGMENT,min_segment_num=$MIN_SEGMENT_NUM,complementary_segment=$COMPLEMENTARY_SEGMENT,token_selection_method=$TOKEN_SELECTION_METHOD,alpha=$ALPHA,temporal_threshold=$TEMPORAL_THRESHOLD,pruning_layer=$PRUNING_LAYER,llm_retention_ratio=$LLM_RETENTION_RATIO"

# Model arguments.
MAX_FRAMES_NUM=32
CONV_TEMPLATE=qwen_1_5
MM_SPATIAL_POOL_MODE=bilinear
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,conv_template=$CONV_TEMPLATE,mm_spatial_pool_mode=$MM_SPATIAL_POOL_MODE,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/anonymous_eval_onevision}"


for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,$BASE_FLASHVID_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
            --main_process_port 18888 \
            --num_processes "$NUM_PROCESSES" \
            -m lmms_eval \
            --model llava_onevision \
            --model_args $MODEL_ARGS \
            --tasks $task \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "llava_onevision" \
            --output_path "$OUTPUT_PATH"
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
