"""
Build images.csv from the train/test JSON splits, enriched with md5 checksums
fetched from the Zenodo public API. The CSV is consumed by cautious-robot:

    cautious-robot -i images.csv -o images/ \
                   --checksum-algorithm md5 --verifier-col md5

Maintenance script — re-run only if the train_test_separate/*.json files change.
The output images.csv is checked into the repo so users do not need network
access until they actually download images.
"""

import csv
import json
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
SPLIT_DIR = DATA_ROOT / "train_test_separate"
OUT_CSV = DATA_ROOT / "images.csv"


def parse_zenodo_url(url):
    """https://zenodo.org/record/<id>/files/<name> -> (id, name)."""
    record, _, name = url.split("/record/")[1].partition("/files/")
    return record, name


def fetch_record_md5s(record_id):
    """Return {filename: md5_hex} for one Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url) as r:
        meta = json.load(r)
    md5s = {}
    for f in meta.get("files", []):
        digest = f.get("checksum", "")
        if digest.startswith("md5:"):
            md5s[f["key"]] = digest[len("md5:"):]
    return md5s


def main():
    entries = {}
    for json_file in sorted(SPLIT_DIR.rglob("*.json")):
        for image_id, url, _mask in json.load(open(json_file)):
            entries.setdefault(image_id, url)

    record_ids = sorted({parse_zenodo_url(url)[0] for url in entries.values()})
    print(f"Fetching md5s for {len(record_ids)} Zenodo records...")
    md5_by_record = {}
    for rid in record_ids:
        md5_by_record[rid] = fetch_record_md5s(rid)
        print(f"  record {rid}: {len(md5_by_record[rid])} files")

    rows = []
    missing = []
    for image_id, url in sorted(entries.items()):
        rid, name = parse_zenodo_url(url)
        md5 = md5_by_record.get(rid, {}).get(name)
        if md5 is None:
            missing.append((image_id, url))
            continue
        ext = Path(url).suffix
        rows.append((f"{image_id}{ext}", url, md5))

    if missing:
        raise SystemExit(
            f"Missing md5 for {len(missing)} files (first 5): {missing[:5]}"
        )

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "file_url", "md5"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
