from argparse import ArgumentParser

import numpy as np
from PIL import Image
import cv2


parser = ArgumentParser()
parser.add_argument("--image_path", type=str)
parser.add_argument("--image_crop_path", type=str)
parser.add_argument("--mask_image_path_out", type=str)
args = parser.parse_args()

img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
img_seg = cv2.imread(args.image_crop_path, cv2.IMREAD_UNCHANGED)

w, h = img_seg.shape[1], img_seg.shape[0]

# Template matching
best_value = None
best_pos = None
for x in range(img.shape[1]):
    if x+w >= img.shape[1]:
        break
    for y in range(img.shape[0]):
        if y+h >= img.shape[0]:
            break
        
        diff = ((img[y:y+h, x:x+w, :3] - img_seg[:, :, :3])**2).sum()
        if best_value is None or diff < best_value:
            best_value = diff
            best_pos = x, y 
            
x, y = best_pos
print(y, x)
img = np.array(Image.open(args.image_path))
img_seg = np.array(Image.open(args.image_crop_path))
assert img_seg.shape[2] == 4, f"Image crop should have 4 channels (RGBA). Image has {img_seg.shape[2]} channels."

mask = img_seg[:, :, 3] == 255

mask_image = np.zeros(img.shape[:2], dtype=np.uint8)
print(x, w, y, h)
mask_image[y:y+h, x:x+w][mask] = 255

Image.fromarray(mask_image).save(args.mask_image_path_out)
