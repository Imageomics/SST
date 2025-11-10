from argparse import ArgumentParser

import numpy as np
from PIL import Image


parser = ArgumentParser()
parser.add_argument("--mask_image_path", type=str)
parser.add_argument("--mask_image_path_out", type=str)
args = parser.parse_args()

img = Image.open(args.mask_image_path).convert("L")
img_arr = np.array(img)
img_arr[img_arr != 255] = 0
img_arr[img_arr == 255] = 5
Image.fromarray(img_arr.astype(np.uint8)).save(args.mask_image_path_out)
