"""
Download all butterfly images from Zenodo URLs in train_test_separate/ JSON files.

Sequential download with 0.5s delay between requests.
Exponential backoff on HTTP 429 (rate limit).
Skips files that already exist on disk.
"""

import json
import os
import time
import requests
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
SPLIT_DIR = DATA_ROOT / "train_test_separate"
IMAGE_DIR = DATA_ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

DELAY = 0.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 5.0  # seconds


def collect_all_entries():
    """Collect (image_id, url, mask_path) from all train/test JSON files."""
    entries = {}
    for species_dir in sorted(SPLIT_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        for json_file in sorted(species_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            for entry in data:
                image_id = entry[0]
                url = entry[1]
                if image_id not in entries:
                    entries[image_id] = url
    return entries


def download_image(image_id, url, session):
    """Download a single image. Returns True on success, False on permanent failure."""
    ext = Path(url).suffix  # .JPG, .CR2, etc.
    out_path = IMAGE_DIR / f"{image_id}{ext}"

    if out_path.exists():
        return True  # already downloaded

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                return True
            elif resp.status_code == 429:
                print(f"  Rate limited on {image_id}, backing off {backoff:.0f}s (attempt {attempt+1})")
                time.sleep(backoff)
                backoff *= 2
                continue
            elif resp.status_code in (403, 404):
                print(f"  PERMANENT FAIL {resp.status_code} for {image_id}: {url}")
                return False
            else:
                print(f"  HTTP {resp.status_code} for {image_id}, retrying...")
                time.sleep(backoff)
                backoff *= 2
                continue
        except requests.exceptions.RequestException as e:
            print(f"  Network error for {image_id}: {e}, retrying...")
            time.sleep(backoff)
            backoff *= 2
            continue

    print(f"  FAILED after {MAX_RETRIES} retries: {image_id}")
    return False


def main():
    entries = collect_all_entries()
    total = len(entries)
    print(f"Found {total} unique images to download")

    session = requests.Session()
    success = 0
    failed = 0
    skipped = 0
    failed_ids = []

    for i, (image_id, url) in enumerate(entries.items()):
        ext = Path(url).suffix
        out_path = IMAGE_DIR / f"{image_id}{ext}"
        if out_path.exists():
            skipped += 1
            continue

        print(f"[{i+1}/{total}] Downloading {image_id} ...", end=" ", flush=True)
        ok = download_image(image_id, url, session)
        if ok:
            success += 1
            print("OK")
        else:
            failed += 1
            failed_ids.append((image_id, url))

        time.sleep(DELAY)

    print(f"\nDone: {success} downloaded, {skipped} skipped (existed), {failed} failed")
    if failed_ids:
        print("\nFailed images:")
        for image_id, url in failed_ids:
            print(f"  {image_id}: {url}")

        # Save failed list for reference
        failed_path = DATA_ROOT / "failed_downloads.json"
        with open(failed_path, "w") as f:
            json.dump(failed_ids, f, indent=2)
        print(f"\nFailed list saved to {failed_path}")


if __name__ == "__main__":
    main()
