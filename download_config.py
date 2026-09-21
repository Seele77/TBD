import os
import json
import shutil
import random
import argparse
from collections import defaultdict

# ============================
# Path configuration
# ============================
DEFAULT_OUTPUT_JSON_PATH = "./data/llava_video_178k/train.json"
DEFAULT_VIDEO_SAVE_DIR = "./data/llava_video_178k/videos"
HF_DATASET_NAME = "lmms-lab/LLaVA-Video-178K"

configs = [
    "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_activitynet",
    "0_30_s_perceptiontest", "0_30_s_nextqa", "30_60_s_academic_v0_1",
    "30_60_s_youtube_v0_1", "30_60_s_activitynet", "30_60_s_perceptiontest",
    "30_60_s_nextqa", "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1",
    "1_2_m_activitynet", "1_2_m_nextqa", "2_3_m_youtube_v0_1",
    "2_3_m_academic_v0_1", "2_3_m_activitynet", "2_3_m_nextqa", "llava_hound"
]


SPLITS = ["caption", "open_ended", "multi_choice"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a LLaVA-format JSON from LLaVA-Video-178K.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON_PATH, help="Output JSON path.")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_SAVE_DIR, help="Root containing extracted videos.")
    parser.add_argument("--dataset", default=HF_DATASET_NAME, help="Hugging Face dataset repository.")
    parser.add_argument("--configs", nargs="+", default=configs, choices=configs, help="Dataset configs to process.")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS, help="Dataset splits to process.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for downstream sampling.")
    return parser.parse_args()


args = parse_args()
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_JSON_PATH = os.path.abspath(args.output)
VIDEO_SAVE_DIR = os.path.abspath(args.video_dir)
HF_DATASET_NAME = args.dataset
configs = args.configs
SPLITS = args.splits
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH) or ".", exist_ok=True)
if args.seed is not None:
    random.seed(args.seed)

llava_format_data = []
global_idx = 0
missing_count = 0
exist_count = 0

for cfg in configs:
    print(f"\nLoading config: {cfg}")
    video_length = "llava_hound" if cfg=="llava_hound" else "_".join(cfg.split("_")[:3])
    
    # Do not specify split so the loader returns a DatasetDict with all splits.
    try:
        dataset_dict = load_dataset(HF_DATASET_NAME, cfg)
    except Exception as e:
        print(f"Error loading config {cfg}: {e}")
        continue

    # Iterate through the three split categories.
    for split_name in SPLITS:
        if split_name not in dataset_dict:
            print(f"Warning: split '{split_name}' not found in {cfg}; skipping.")
            continue
            
        dataset = dataset_dict[split_name]
        print(f"Processing {cfg} [{split_name}] - {len(dataset)} samples")

        for sample in tqdm(dataset, desc=f"Progress"):
            video_path = sample.get("video", None)

            if video_path is None:
                missing_count += 1
                continue

            # Actual local video path.
            real_video_path = os.path.join(VIDEO_SAVE_DIR, video_path)

            # Check whether the video file exists locally.
            if not os.path.exists(real_video_path):
                missing_count += 1
                continue

            exist_count += 1
            
            # Handle multiple possible field names.
            conversations = []
            if "conversations" in sample:
                conversations = sample["conversations"]
            elif "messages" in sample:
                conversations = sample["messages"]

            llava_sample = {
                "id": str(global_idx),
                "video": video_path,
                "conversations": conversations,
                "video_length": video_length,
                "split": split_name
            }

            llava_format_data.append(llava_sample)
            global_idx += 1

print("\nVideo check finished")
print(f"Existing videos: {exist_count}")
print(f"Missing videos: {missing_count}")

# random.seed(42)
# llava_format_data = random.sample(llava_format_data, 50000)

# ============================
# Statistics
# ============================
sampled_length_counter = defaultdict(int)
sampled_split_counter = defaultdict(int)
for sample in llava_format_data:
    sampled_length_counter[sample["video_length"]] += 1
    sampled_split_counter[sample["split"]] += 1

print("\nVideo length distribution:")
for k in sorted(sampled_length_counter.keys()):
    print(f"{k}: {sampled_length_counter[k]}")

print("\nSplit distribution:")
for k in sampled_split_counter:
    print(f"{k}: {sampled_split_counter[k]}")

# ============================
# Save merged JSON
# ============================
print("\nSaving merged JSON...")
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(llava_format_data, f, ensure_ascii=False, indent=2)  # Keep indentation for readability.

print("Done.")
print(f"Total samples merged: {len(llava_format_data)}")
print(f"JSON saved to: {OUTPUT_JSON_PATH}")
