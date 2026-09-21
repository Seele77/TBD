# Token-Budget Distillation (TBD)

Official implementation of **Token-Budget Distillation: Transferring Full-Token Semantics to Compressed Video Vision-Language Models**.

Paper: [arXiv:2608.28138](https://arxiv.org/abs/2608.28138)

TBD trains LoRA adapters for compressed video VLMs through a full-token teacher and a FlashVID-compressed student. This repository provides training and evaluation support for LLaVA-Video and LLaVA-OneVision.

## Requirements

The reference environment is Linux, Python 3.10, CUDA 12.1, PyTorch 2.1.2, DeepSpeed 0.14.4, and FlashAttention 2.5.7.

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
python -m pip install -e FlashVID/lmms-eval
python -m pip install -e FlashVID --no-deps
```

## Data preparation

TBD uses the LLaVA-Video-178K instruction data. The helper scripts accept command-line paths and are safe to resume:

```bash
python download_videos.py \
  --video-dir /data/llava_video_178k/videos \
  --tmp-dir /data/llava_video_178k/tmp

python download_config.py \
  --output /data/llava_video_178k/train.json \
  --video-dir /data/llava_video_178k/videos
```

Use `--configs` and `--max-shards` to select a subset or limit downloads.

## Training

Reproduce the main LLaVA-Video experiment with:

```bash
DATA_PATH=/data/llava_video_178k/train.json \
VIDEO_FOLDER=/data/llava_video_178k/videos \
OUTPUT_DIR=./outputs/tbd_llava_video_r10 \
bash scripts/flashvid_kd/run_llava_video_flashvid_lora_kd.sh
```

For the LLaVA-OneVision variant, run:

```bash
DATA_PATH=/data/llava_video_178k/train.json \
VIDEO_FOLDER=/data/llava_video_178k/videos \
OUTPUT_DIR=./outputs/tbd_onevision_r10 \
bash scripts/flashvid_kd/run_llava_onevision_flashvid_lora_kd.sh
```

Set `CUDA_VISIBLE_DEVICES` before launching DeepSpeed to choose the visible GPUs. The complete training configuration is in the launcher scripts.

## Merge and evaluate

Merge the LoRA adapter and non-LoRA trainables into a standalone checkpoint:

```bash
MODEL_NAME=lmms-lab/LLaVA-Video-7B-Qwen2 \
OUTPUT_DIR=./outputs/tbd_llava_video_r10 \
MERGED_OUTPUT_DIR=./outputs/tbd_llava_video_r10_merged \
bash scripts/flashvid_kd/merge_lora.sh
```

Evaluation uses the bundled `lmms-eval` fork:

```bash
PRETRAINED=./outputs/tbd_llava_video_r10_merged \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
TASKS='videomme mvbench egoschema longvideobench_val_v' \
bash FlashVID/scripts/llava_vid.sh
```

Use `FlashVID/scripts/llava_ov.sh` for OneVision. Set `PRETRAINED`, `CUDA_VISIBLE_DEVICES`, `NUM_PROCESSES`, `TASKS`, and `OUTPUT_PATH` as needed. Results are written to `logs/` by default.

## Citation

If this repository is useful in your research, please cite the [arXiv paper](https://arxiv.org/abs/2608.28138):

```bibtex
@misc{guo2026tokenbudgetdistillationtransferringfulltoken,
  title        = {Token-Budget Distillation: Transferring Full-Token Semantics to Compressed Video Vision-Language Models},
  author       = {Xiaoyang Guo and Guoping Luo and Jusheng Zhang and Keze Wang and Wenhao Wang},
  year         = {2026},
  eprint       = {2608.28138},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2608.28138}
}
```

## License and attribution

This repository combines original TBD code with components derived from LLaVA and FlashVID. LLaVA-derived files retain the Apache-2.0 notices in their source headers; the FlashVID component is distributed under the MIT license in [`FlashVID/LICENSE`](FlashVID/LICENSE). Checkpoint, benchmark, and dataset terms are controlled by their respective upstream providers. Preserve all upstream notices when redistributing a modified copy.
