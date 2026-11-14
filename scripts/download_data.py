"""
Data Download and Preparation Script for TinyStories.

Downloads TinyStoriesV2-GPT4:
- The first ~300 MB of TinyStoriesV2-GPT4-train.txt using HTTP Range requests.
  Aligns to the last '<|endoftext|>' separator within the range.
- The full TinyStoriesV2-GPT4-valid.txt (~22.5 MB).
Saves files to data/raw/train.txt and data/raw/valid.txt.
Prints statistics and first 3 stories.
"""

import os
import sys
import httpx
from tqdm import tqdm

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

RAW_DIR = "data/raw"
TRAIN_DEST = os.path.join(RAW_DIR, "train.txt")
VALID_DEST = os.path.join(RAW_DIR, "valid.txt")
TARGET_TRAIN_BYTES = 300 * 1024 * 1024  # 300 MB


def download_range(url: str, dest_path: str, max_bytes: int):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        print(f"{dest_path} already exists ({os.path.getsize(dest_path):,} bytes). Skipping download.")
        return

    print(f"Downloading first {max_bytes / (1024 * 1024):.0f} MB of {url}...")
    headers = {"Range": f"bytes=0-{max_bytes - 1}"}
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code not in (200, 206):
                raise RuntimeError(f"HTTP error {response.status_code} while downloading {url}")
            total = int(response.headers.get("content-length", max_bytes))
            content_chunks = []
            with tqdm(total=total, unit="B", unit_scale=True, desc="Train data") as pbar:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    content_chunks.append(chunk)
                    pbar.update(len(chunk))
    raw_data = b"".join(content_chunks)
    # Align to the last complete story
    sep = b"<|endoftext|>"
    last_idx = raw_data.rfind(sep)
    if last_idx != -1:
        raw_data = raw_data[: last_idx + len(sep)]
        print(f"Aligned training data to last complete story separator at {len(raw_data):,} bytes.")
    with open(dest_path, "wb") as f:
        f.write(raw_data)
    print(f"Saved {len(raw_data):,} bytes to {dest_path}")


def download_full(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        print(f"{dest_path} already exists ({os.path.getsize(dest_path):,} bytes). Skipping download.")
        return

    print(f"Downloading full {url}...")
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP error {response.status_code} while downloading {url}")
            total = int(response.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc="Validation data") as pbar:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(f"Saved {os.path.getsize(dest_path):,} bytes to {dest_path}")


def analyze_file(path: str, name: str):
    print(f"\n--- Statistics for {name} ({path}) ---")
    size_bytes = os.path.getsize(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    chars = len(text)
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    num_stories = len(stories)
    avg_len = chars / num_stories if num_stories else 0
    print(f"File Size:            {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"Total Characters:     {chars:,}")
    print(f"Total Stories:        {num_stories:,}")
    print(f"Average Story Length: {avg_len:.1f} characters")
    return stories


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    download_range(TRAIN_URL, TRAIN_DEST, TARGET_TRAIN_BYTES)
    download_full(VALID_URL, VALID_DEST)

    train_stories = analyze_file(TRAIN_DEST, "Train Split")
    val_stories = analyze_file(VALID_DEST, "Validation Split")

    print("\n" + "=" * 60)
    print("FIRST 3 STORIES (Train Split):")
    print("=" * 60)
    for i, story in enumerate(train_stories[:3], 1):
        print(f"\n[Story {i}]:\n{story}")
        print("-" * 40)


if __name__ == "__main__":
    main()
