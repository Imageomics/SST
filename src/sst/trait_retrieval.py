"""Rank query images by how well a support trait mask cycles back to itself through each query.

For every query image the support mask is propagated to the query and then propagated back to the
support frame. The intersection over union between the round tripped mask and the original support
mask scores how consistently the trait is present, and the queries are ranked by that score.
"""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sst import sam_utils


def add_arguments(parser):
    parser.add_argument("--support_image", type=str, required=True, help="Path to the support image.")
    parser.add_argument("--support_mask", type=str, required=True, help="Path to the support segmentation mask.")
    parser.add_argument("--trait_id", type=int, required=True, help="Trait id to retrieve (its value in the mask).")
    parser.add_argument("--query_images", type=str, required=True, help="Path to the query images folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--output_format", choices=["png", "gif"], default="gif", help="Output format.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top retrievals to save.")
    parser.add_argument("--model", type=str, default=sam_utils.DEFAULT_SAM2_MODEL, help="SAM2 model id.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (default: auto).")
    return parser


def cycle_consistency(tracker, orig_image, target_image, masks_list, return_vis=False):
    """Score each mask in ``masks_list`` by round tripping it through the target image.

    Propagates the original masks to the target, then propagates the inferred target masks back to
    the original frame, and returns the per object IoU between the round tripped masks and the
    originals. When ``return_vis`` is set, also returns the target image with the inferred mask
    contours drawn on it.
    """
    inferred = tracker.segment(orig_image, masks_list, [target_image])[-1]

    vises = None
    if return_vis:
        vises = []
        for mask in inferred["segmentation"]:
            mask = cv2.resize(mask.astype(np.uint8), (target_image.shape[1], target_image.shape[0]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vises.append(cv2.drawContours(target_image.copy(), contours, -1, (255, 0, 0), 10))

    inferred_masks = list(inferred["segmentation"])
    reconstructed = tracker.segment(orig_image, inferred_masks, [target_image])[-1]

    iou_records = np.zeros(len(masks_list))
    for obj_id, mask in zip(reconstructed["obj_ids"], reconstructed["segmentation"]):
        orig_mask = np.asarray(masks_list[obj_id], dtype=bool)
        if mask.shape != orig_mask.shape:
            mask = cv2.resize(mask.astype(np.uint8), (orig_mask.shape[1], orig_mask.shape[0])).astype(bool)
        intersection = np.logical_and(mask, orig_mask).sum()
        union = np.logical_or(mask, orig_mask).sum()
        iou_records[obj_id] = intersection / union if union else 0.0

    return (iou_records, vises) if return_vis else iou_records


def run(args):
    print("Loading support image and mask...")
    support_image = cv2.imread(args.support_image)[..., ::-1]
    support_mask = cv2.imread(args.support_mask, cv2.IMREAD_GRAYSCALE)
    support_mask = (support_mask == args.trait_id).astype(np.uint8)

    query_names = sorted(os.listdir(args.query_images))
    query_images = [cv2.imread(os.path.join(args.query_images, name))[..., ::-1] for name in query_names]

    tracker = sam_utils.Sam2Tracker(model_id=args.model, device=args.device)

    print("Retrieving images...")
    retrieve_scores = []
    retrieve_vis = []
    for query_image in tqdm(query_images):
        iou_records, vises = cycle_consistency(tracker, support_image, query_image, [support_mask], return_vis=True)
        retrieve_scores.append(iou_records[0])
        retrieve_vis.append(vises[0])

    # Preserve the original ordering: query frames sorted by ascending cycle-consistency score.
    sorted_indices = np.argsort(retrieve_scores)
    sorted_images = [retrieve_vis[i] for i in sorted_indices]

    os.makedirs(args.output, exist_ok=True)
    if args.output_format == "gif":
        print(f"Saving the top-{args.top_k} retrieved images as a gif...")
        gif_images = [Image.fromarray(img) for img in sorted_images[: args.top_k]]
        min_size = sorted(gif_images, key=lambda x: x.size)[0].size
        gif_images = [img.resize(min_size) for img in gif_images]
        gif_images[0].save(
            os.path.join(args.output, f"top_{args.top_k}_retrieved.gif"),
            save_all=True,
            append_images=gif_images[1:],
            duration=1000,
            loop=0,
        )
    else:
        print(f"Saving the top-{args.top_k} retrieved images as png...")
        for i, img in enumerate(sorted_images[: args.top_k]):
            cv2.imwrite(os.path.join(args.output, f"retrieved_rank_{i + 1}.png"), img[..., ::-1])

    print("Done! Retrieval results saved at", args.output)
