"""
Parallel butterfly image downloader.

Uses concurrent.futures with multiple workers to download from Zenodo.
Skips files that already exist. Retries on failure with backoff.
"""

import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
SPLIT_DIR = DATA_ROOT / "train_test_separate"
IMAGE_DIR = DATA_ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

NUM_WORKERS = 16
MAX_RETRIES = 3
TIMEOUT = 120


def collect_all_entries():
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


def download_one(args):
    image_id, url = args
    ext = Path(url).suffix
    out_path = IMAGE_DIR / f"{image_id}{ext}"

    if out_path.exists():
        return image_id, "skipped"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                return image_id, "ok"
            elif resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            elif resp.status_code in (403, 404):
                return image_id, f"fail_{resp.status_code}"
            else:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            time.sleep(2 ** attempt)
            continue

    return image_id, "fail_retry"


def main():
    entries = collect_all_entries()
    total = len(entries)

    # Filter to only those not yet downloaded
    todo = []
    skipped = 0
    for image_id, url in entries.items():
        ext = Path(url).suffix
        if (IMAGE_DIR / f"{image_id}{ext}").exists():
            skipped += 1
        else:
            todo.append((image_id, url))

    print(f"Total: {total}, already downloaded: {skipped}, remaining: {len(todo)}")
    print(f"Using {NUM_WORKERS} parallel workers")

    if not todo:
        print("Nothing to download!")
        return

    success = 0
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_one, item): item for item in todo}
        for i, future in enumerate(as_completed(futures)):
            image_id, status = future.result()
            if status == "ok":
                success += 1
            elif status != "skipped":
                failed.append((image_id, futures[future][1], status))

            done = i + 1
            if done % 100 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(todo)}] {success} ok, {len(failed)} failed, "
                      f"{rate:.1f} img/s, ETA {eta/60:.0f}m")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m: {success} downloaded, {skipped} existed, {len(failed)} failed")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for img_id, url, status in failed:
            print(f"  {img_id}: {status} — {url}")
        with open(DATA_ROOT / "failed_downloads.json", "w") as f:
            json.dump([(i, u, s) for i, u, s in failed], f, indent=2)


if __name__ == "__main__":
    main()
