import os
import tarfile
import argparse

# ============================
# Path configuration
# ============================
DEFAULT_VIDEO_SAVE_DIR = "./data/llava_video_178k/videos"
DEFAULT_TMP_DOWNLOAD_DIR = "./data/llava_video_178k/tmp"


configs = [
    "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_activitynetqa",
    "0_30_s_perceptiontest", "0_30_s_nextqa", "30_60_s_academic_v0_1",
    "30_60_s_youtube_v0_1", "30_60_s_activitynetqa", "30_60_s_perceptiontest",
    "30_60_s_nextqa", "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1",
    "1_2_m_activitynetqa", "1_2_m_nextqa", "2_3_m_youtube_v0_1",
    "2_3_m_academic_v0_1", "2_3_m_activitynetqa", "2_3_m_nextqa", "llava_hound"
]



# ============================
# tar.gz naming rule
# ============================
# Video shards on HuggingFace usually follow: {config}_videos_*.tar.gz
# Increase this cap if a config has more shards.
MAX_TARS_PER_CONFIG = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract LLaVA-Video-178K video shards.")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_SAVE_DIR, help="Directory for extracted videos.")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DOWNLOAD_DIR, help="Directory for downloaded archives.")
    parser.add_argument("--max-shards", type=int, default=MAX_TARS_PER_CONFIG, help="Maximum shards to probe per config.")
    parser.add_argument("--configs", nargs="+", default=configs, choices=configs, help="Dataset configs to download.")
    return parser.parse_args()


def safe_extract(tar, destination):
    destination = os.path.abspath(destination)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(destination, member.name))
        if os.path.commonpath((destination, target)) != destination:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    tar.extractall(path=destination)

# ============================
# Start download + extraction
# ============================
args = parse_args()
from huggingface_hub import hf_hub_download

VIDEO_SAVE_DIR = os.path.abspath(args.video_dir)
TMP_DOWNLOAD_DIR = os.path.abspath(args.tmp_dir)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
os.makedirs(TMP_DOWNLOAD_DIR, exist_ok=True)

for cfg in args.configs:
    print(f"\nProcessing config: {cfg}")
    
    for idx in range(1, args.max_shards + 1):
        tar_filename = f"{cfg}_videos_{idx}.tar.gz"
        hf_path = f"{cfg}/{tar_filename}"

        local_tar_path = os.path.join(TMP_DOWNLOAD_DIR, cfg)
        local_tar_path = os.path.join(local_tar_path, tar_filename)
        
        if not os.path.exists(local_tar_path):
            try:
                print(f"Downloading {hf_path} ...")
                hf_hub_download(
                    repo_id="lmms-lab/LLaVA-Video-178K",
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir=TMP_DOWNLOAD_DIR,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Warning: {hf_path} not found or failed: {e}")
                continue  # Move to the next tar shard.
        else:
            print(f"{tar_filename} already downloaded. Skipping.")
            continue

        # Extract tar.gz
        extracted_dir = os.path.join(VIDEO_SAVE_DIR, f"{cfg}_videos_{idx}")
        if not os.path.exists(extracted_dir):
            print(f"Extracting {tar_filename} ...")
            with tarfile.open(local_tar_path, "r:gz") as tar:
                safe_extract(tar, VIDEO_SAVE_DIR)
        else:
            print(f"{tar_filename} already extracted. Skipping.")

print("\nAll done. Videos are in:", VIDEO_SAVE_DIR)
