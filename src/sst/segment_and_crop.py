"""Segment specimens against a support mask and crop each query image to the segmented region.

Two modes share the same cropping logic. The default batch mode loads a flat folder of query images
and tracks them in a single video session. ``--per-image`` walks a directory tree, segments one image
at a time (which keeps memory bounded and supports resuming), and can read raw formats such as CR2.
"""

import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sst import sam_utils

RAW_EXTENSIONS = (".cr2", ".tiff", ".tif", ".nef", ".arw", ".dng")


def add_arguments(parser):
    parser.add_argument("--support_image", type=str, required=True, help="Path to the support image.")
    parser.add_argument("--support_mask", type=str, required=True, help="Path to the support segmentation mask.")
    parser.add_argument("--query_images", type=str, required=True, help="Path to the query images folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--model", type=str, default=sam_utils.DEFAULT_SAM2_MODEL, help="SAM2 model id.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (default: auto).")
    parser.add_argument("--per-image", dest="per_image", action="store_true",
                        help="Walk the folder recursively and process one image at a time (supports raw formats).")
    parser.add_argument("--no-reprocess", dest="no_reprocess", action="store_true",
                        help="In per-image mode, skip query images whose output already exists.")
    return parser


def _load_image(path):
    """Load an image as an RGB array, decoding raw formats with rawpy when needed."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise ImportError(
                f"Reading {ext} files requires the optional 'raw' extra. Install it with "
                "'pip install sstrack[raw]'."
            ) from exc
        with rawpy.imread(path) as raw:
            return raw.postprocess()
    return cv2.imread(path)[..., ::-1]


def _crop_to_mask(query_img, segmentation):
    """Black out everything outside the union of the masks and crop to its bounding box."""
    combined = np.zeros(query_img.shape[:2], dtype=bool)
    for mask in segmentation:
        mask = cv2.resize(mask.astype(np.uint8), (query_img.shape[1], query_img.shape[0]))
        combined |= mask.astype(bool)

    masked = query_img.copy()
    masked[~combined] = 0

    rows = np.where(combined.any(axis=1))[0]
    cols = np.where(combined.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return masked
    return masked[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1, :]


def _load_support(args):
    support_image = cv2.imread(args.support_image)[..., ::-1]
    support_mask = cv2.imread(args.support_mask, cv2.IMREAD_GRAYSCALE)
    support_masks = [support_mask == i for i in range(1, support_mask.max() + 1)]
    return support_image, support_masks


def _run_batch(args, tracker, support_image, support_masks):
    query_names = sorted(os.listdir(args.query_images))
    query_paths = [os.path.join(args.query_images, name) for name in query_names]
    query_images = [cv2.imread(path)[..., ::-1] for path in query_paths]

    print("Inferring the masks...")
    frames_info = tracker.segment(support_image, support_masks, query_images, verbose=True)

    print("Saving results...")
    for query_path, query_img, info in tqdm(
        list(zip(query_paths, query_images, frames_info[1:])), desc="Saving images"
    ):
        cropped = _crop_to_mask(query_img, info["segmentation"])
        name = os.path.splitext(os.path.basename(query_path))[0]
        Image.fromarray(cropped).save(os.path.join(args.output, f"{name}.png"))


def _run_per_image(args, tracker, support_image, support_masks):
    query_paths = []
    for root, _dirs, files in os.walk(args.query_images):
        for name in files:
            query_paths.append(os.path.join(root, name))
    random.shuffle(query_paths)

    print("Inferring the masks...")
    for path in tqdm(query_paths, desc="Processing and saving"):
        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(args.output, f"{name}.png")
        if args.no_reprocess and os.path.exists(save_path):
            continue
        query_img = _load_image(path)
        frames_info = tracker.segment(support_image, support_masks, [query_img], verbose=False)
        cropped = _crop_to_mask(query_img, frames_info[1]["segmentation"])
        Image.fromarray(cropped).save(save_path)


def run(args):
    print("Loading support image and mask...")
    support_image, support_masks = _load_support(args)
    os.makedirs(args.output, exist_ok=True)

    tracker = sam_utils.Sam2Tracker(model_id=args.model, device=args.device)
    if args.per_image:
        _run_per_image(args, tracker, support_image, support_masks)
    else:
        _run_batch(args, tracker, support_image, support_masks)

    print("Done! The output is saved in", args.output)
