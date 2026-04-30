"""
Flatten train/test JSON splits into a single CSV consumable by cautious-robot.

Each row: filename (= "<image_id>.<ext>"), file_url.
Run once before `cautious-robot -i images.csv -o images/`.
"""

import csv
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
SPLIT_DIR = DATA_ROOT / "train_test_separate"
OUT_CSV = DATA_ROOT / "images.csv"


def main():
    seen = {}
    for species_dir in sorted(SPLIT_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        for json_file in sorted(species_dir.glob("*.json")):
            with open(json_file) as f:
                for image_id, url, _mask in json.load(f):
                    if image_id in seen:
                        continue
                    ext = Path(url).suffix
                    seen[image_id] = (f"{image_id}{ext}", url)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "file_url"])
        for filename, url in seen.values():
            w.writerow([filename, url])

    print(f"Wrote {len(seen)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
