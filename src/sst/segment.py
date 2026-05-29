"""Propagate a support mask across a folder of query images and save the visualized segmentations."""

import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sst import sam_utils


def add_arguments(parser):
    parser.add_argument("--support_image", type=str, required=True, help="Path to the support image.")
    parser.add_argument("--support_mask", type=str, required=True, help="Path to the support segmentation mask.")
    parser.add_argument("--query_images", type=str, required=True, help="Path to the query images folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--output_format", choices=["png", "gif"], default="gif", help="Output format.")
    parser.add_argument("--model", type=str, default=sam_utils.DEFAULT_SAM2_MODEL, help="SAM2 model id.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (default: auto).")
    return parser


def run(args):
    print("Loading support image and mask...")
    support_image = cv2.imread(args.support_image)[..., ::-1]
    support_mask = cv2.imread(args.support_mask, cv2.IMREAD_GRAYSCALE)
    support_masks = [support_mask == i for i in range(1, support_mask.max() + 1)]

    query_names = sorted(os.listdir(args.query_images))
    query_images = [cv2.imread(os.path.join(args.query_images, name))[..., ::-1] for name in query_names]

    tracker = sam_utils.Sam2Tracker(model_id=args.model, device=args.device)

    print("Inferring the masks...")
    frames_info = tracker.segment(support_image, support_masks, query_images, verbose=True)

    frames = [support_image] + query_images
    print("Visualizing the results...")
    output_imgs = []
    for frame, info in zip(frames, frames_info):
        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        for obj_id, mask in zip(info["obj_ids"], info["segmentation"]):
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
            sam_utils.show_mask(mask, plt.gca(), obj_id, borders=True, alpha=0.75)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        output_imgs.append(Image.open(buf))

    os.makedirs(args.output, exist_ok=True)
    if args.output_format == "gif":
        output_imgs[0].save(
            os.path.join(args.output, "out.gif"),
            save_all=True,
            append_images=output_imgs[1:],
            loop=0,
            duration=1000,
        )
    else:
        for i, img in enumerate(output_imgs):
            img.save(os.path.join(args.output, f"{i:06d}.png"))

    print("Done! The output is saved in", args.output)
