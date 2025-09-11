import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import sam_utils
import cv2
import io
import argparse
from PIL import Image
from tqdm import tqdm
import rawpy

# parse the arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--support_image', type=str, help='Path to the support image.')
parser.add_argument('--support_mask', type=str, help='Path to the support segmentation mask.')
parser.add_argument('--query_images', type=str, help='Path to the query images folder.')
parser.add_argument('--output', type=str, help='Path to the output folder.')
parser.add_argument('--do_not_reprocess', action='store_true', help='Do not reprocess already processed images')

args = parser.parse_args()
support_image_path = args.support_image
support_mask_path = args.support_mask
query_images_folder = args.query_images
output_folder = args.output

# load the support image and mask
print ("Loading support image and mask...")
support_image = cv2.imread(support_image_path)[..., ::-1]
support_mask = cv2.imread(support_mask_path, cv2.IMREAD_GRAYSCALE)
support_masks = [support_mask == i for i in range(1, support_mask.max()+1)]

# load the query images

query_image_paths = []
for root, dirs, files in os.walk(query_images_folder):
    for f in files:
        query_image_paths.append(os.path.join(root, f))
#query_images = [cv2.imread(path)[..., ::-1] for path in query_image_paths]

# build the predictor
video_predictor = sam_utils.build_sam2_predictor()

# Make output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# Shuffle paths
random.seed(time.time())
random.shuffle(query_image_paths)
    
# load the support image and mask
print ("Inferring the masks...")
for path in tqdm(query_image_paths, desc="processing and saving"):
    name, ext = os.path.splitext(os.path.basename(path))
    image_save_path = os.path.join(output_folder, f"{name}.png")
    if args.do_not_reprocess and os.path.exists(image_save_path):
        continue
    
    if ext.lower() in [".cr2", ".tiff"]:
        raw = rawpy.imread(path)
        rgb = raw.postprocess()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)[..., ::-1]
    state = sam_utils.load_masks(video_predictor, [img], support_image, support_masks, verbose=False)
    frames_info = sam_utils.propagate_masks(video_predictor, state, verbose=False)
        
    frame = frames_info[1] # skip template frame
    
    # Create masked image where only mask regions are visible
    query_img = img.copy()
    out_masks = frame['segmentation']
    out_masks = [cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0])) for mask in out_masks]
    obj_ids = frame['obj_ids']
    
    # Create combined mask from all detected objects
    combined_mask = np.zeros((query_img.shape[0], query_img.shape[1]), dtype=bool)
    for mask in out_masks:
        combined_mask = np.logical_or(combined_mask, mask.astype(bool))
    
    # Apply mask: set non-mask regions to black
    masked_img = query_img.copy()
    masked_img[~combined_mask] = [0, 0, 0]  # Set non-mask pixels to black
    
    ymin, ymax = None, None
    xmin, xmax = None, None
    for y in range(masked_img.shape[0]):
        v = masked_img[y, :, :].sum()
        if v > 0:
            ymin = y
            break
            
    for y in range(masked_img.shape[0]-1, -1, -1):
        v = masked_img[y, :, :].sum()
        if v > 0:
            ymax = y
            break
    
    for x in range(masked_img.shape[1]):
        v = masked_img[:, x, :].sum()
        if v > 0:
            xmin = x
            break
            
    for x in range(masked_img.shape[1]-1 , -1, -1):
        v = masked_img[:, x, :].sum()
        if v > 0:
            xmax = x
            break
    
    #print(f"Frame {i}: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}")
    masked_img = masked_img[ymin:ymax, xmin:xmax, :]
    
    
    Image.fromarray(masked_img).save(image_save_path)


print ("Done! The output is saved in", output_folder)

