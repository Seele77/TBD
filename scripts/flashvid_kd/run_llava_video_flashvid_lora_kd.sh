#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-lmms-lab/LLaVA-Video-7B-Qwen2}
DATA_PATH=${DATA_PATH:-./data/llava_video_178k/train.json}
VIDEO_FOLDER=${VIDEO_FOLDER:-./data/llava_video_178k/videos}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/anonymous_experiment_video}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-./scripts/zero2.json}
FRAMES_UPBOUND=${FRAMES_UPBOUND:-32}
VIDEO_FPS=${VIDEO_FPS:-1}

deepspeed llava/train/train.py \
  --deepspeed ${DEEPSPEED_CFG} \
  --model_name_or_path ${MODEL_NAME} \
  --version qwen_1_5 \
  --data_path ${DATA_PATH} \
  --video_folder ${VIDEO_FOLDER} \
  --vision_tower google/siglip-so400m-patch14-384 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio anyres \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 4 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --frames_upbound ${FRAMES_UPBOUND} \
  --video_fps ${VIDEO_FPS} \
  --dataloader_num_workers 0 \
  --report_to none \
  --attn_implementation flash_attention_2 \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_include_mm_projector True \
  --enable_flashvid True \
  --flashvid_retention_ratio 0.1 \
  --flashvid_do_segment True \
  --flashvid_segment_threshold 0.9 \
  --flashvid_min_segment_num 8 \
  --flashvid_complementary_segment True \
  --flashvid_token_selection_method attn_div_v2 \
  --flashvid_alpha 0.7 \
  --flashvid_temporal_threshold 0.8 \
  --flashvid_expansion 1.25 \
  --flashvid_pruning_layer 20 \
  --flashvid_enable_llm_compression False \
  --dualpath_kd True \
  --kd_loss_weight 1.0 \
  --ce_loss_weight 1.0 \
  --kd_temperature 1.0 \
  --kd_warmup_ratio 0.1 \
  --kd_topk 0 \
  --kd_teacher_confidence_threshold 0.3 \
  --kd_only_teacher_correct True \
  --kd_gt_confidence_threshold 0.2 \
  --kd_max_ce_ratio 0.6 \
  --kd_margin_loss_weight 0.15 \
  --kd_margin_huber_delta 1.0 \
  --teacher_disable_lora True
