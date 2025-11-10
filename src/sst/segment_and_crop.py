import os
import numpy as np
import matplotlib.pyplot as plt
import sam_utils
import cv2
import io
import argparse
from PIL import Image
from tqdm import tqdm

# parse the arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--support_image', type=str, help='Path to the support image.')
parser.add_argument('--support_mask', type=str, help='Path to the support segmentation mask.')
parser.add_argument('--query_images', type=str, help='Path to the query images folder.')
parser.add_argument('--output', type=str, help='Path to the output folder.')

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
query_image_paths = [os.path.join(query_images_folder, img) for img in sorted(os.listdir(query_images_folder))]
query_images = [cv2.imread(path)[..., ::-1] for path in query_image_paths]

# build the predictor
video_predictor = sam_utils.build_sam2_predictor()

# Make output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# load the support image and mask
print ("Inferring the masks...")
state = sam_utils.load_masks(video_predictor, query_images, support_image, support_masks, verbose=True)
frames_info = sam_utils.propagate_masks(video_predictor, state, verbose=True)


# visualize the results
output_imgs = []
print ("Saving results...")
for i, frame in tqdm(enumerate(frames_info), total=len(query_image_paths)+1, desc="Saving Images"):
    if i == 0: # skip template frame
        continue
    
    # Create masked image where only mask regions are visible
    query_img = query_images[i].copy()
    out_masks = frame['segmentation']
    out_masks = [cv2.resize(mask.astype(np.uint8), (query_images[i].shape[1], query_images[i].shape[0])) for mask in out_masks]
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
    
    img_path = query_image_paths[i-1]
    name = os.path.splitext(os.path.basename(img_path))[0]
    Image.fromarray(masked_img).save(os.path.join(output_folder, f"{name}.png"))


print ("Done! The output is saved in", output_folder)

