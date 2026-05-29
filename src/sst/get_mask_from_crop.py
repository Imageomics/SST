"""Locate a transparent crop within its source image and emit the corresponding full-image mask.

The crop is matched against the source with template matching using its alpha channel as the
template mask, and the crop's alpha is then written back into a full-size mask at the matched
location.
"""

import cv2
import numpy as np
from PIL import Image


def add_arguments(parser):
    parser.add_argument("--image_path", type=str, required=True, help="Path to the full source image.")
    parser.add_argument("--image_crop_path", type=str, required=True, help="Path to the RGBA crop.")
    parser.add_argument("--mask_image_path_out", type=str, required=True, help="Output mask path.")
    return parser


def run(args):
    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_seg = cv2.imread(args.image_crop_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    seg_mask = np.uint8(img_seg[:, :, 3] != 0)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    w, h = img_seg.shape[1], img_seg.shape[0]
    res = cv2.matchTemplate(img, img_seg, cv2.TM_SQDIFF, mask=seg_mask)
    loc = np.where(res == res.min())
    y, x = int(loc[0][0]), int(loc[1][0])

    img = np.array(Image.open(args.image_path))
    img_seg = np.array(Image.open(args.image_crop_path))
    assert img_seg.shape[2] == 4, f"Image crop should have 4 channels (RGBA). Image has {img_seg.shape[2]} channels."

    mask = img_seg[:, :, 3] > 0
    mask_image = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_image[y:y + h, x:x + w][mask] = 255
    Image.fromarray(mask_image).save(args.mask_image_path_out)
