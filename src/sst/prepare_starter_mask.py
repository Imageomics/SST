"""Convert a white-on-black mask image into the integer object-id mask the tracker expects."""

import numpy as np
from PIL import Image


def add_arguments(parser):
    parser.add_argument("--mask_image_path", type=str, required=True, help="Input mask image (white foreground).")
    parser.add_argument("--mask_image_path_out", type=str, required=True, help="Output object-id mask path.")
    return parser


def run(args):
    img = Image.open(args.mask_image_path).convert("L")
    arr = np.array(img)
    arr[arr != 255] = 0
    arr[arr == 255] = 5
    Image.fromarray(arr.astype(np.uint8)).save(args.mask_image_path_out)
