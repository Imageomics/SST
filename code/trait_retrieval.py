#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import sam_utils
import cv2
import io
import argparse
from PIL import Image
from tqdm import tqdm

def cycle_consistency(orig_image, target_image, masks_list, predictor, return_vis = False):
    '''
    This function calculates the reconstruction performance of masks using cycle consistency.
    orig_image: numpy array, the original image.
    target_image: numpy array, the target image.
    masks_list: list of dictionaries, the list of masks.
    predictor: the video tracker.
    '''
    # return a list of reconstruction performance for masks
    orig_image = cv2.resize(orig_image, (1024, 1024))
    target_image = cv2.resize(target_image, (1024, 1024))
    state = sam_utils.load_masks(predictor, [target_image], orig_image, masks_list, offload_state_to_cpu=False, offload_video_to_cpu=False)
    frames_info = sam_utils.propagate_masks(predictor, state, verbose=False)
    inferred_masks = frames_info[-1]
    if return_vis:
        vises = []
        for obj_id, mask in zip(inferred_masks['obj_ids'], inferred_masks['segmentation']):
            if mask.shape != (1024, 1024):
                mask = cv2.resize(mask.astype(np.uint8), (1024, 1024))
            vis_img = target_image.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis = cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 10)
            vises.append(vis)
    else:
        vises = None
    inferred_masks = [mask for mask in inferred_masks['segmentation']]
    state = sam_utils.load_masks(predictor, [target_image], orig_image, inferred_masks, offload_state_to_cpu=False, offload_video_to_cpu=False)
    frames_info = sam_utils.propagate_masks(predictor, state, verbose=False)
    reconstructed_masks = frames_info[-1]
    iou_records = np.zeros(len(masks_list))
    # compare the reconstructed masks with the original masks
    for obj_id, mask in zip(reconstructed_masks['obj_ids'], reconstructed_masks['segmentation']):
        if mask.shape != (1024, 1024):
            mask = cv2.resize(mask.astype(np.uint8), (1024, 1024))
            mask = mask.astype(bool)
        orig_mask = masks_list[obj_id]
        if orig_mask.shape != (1024, 1024):
            orig_mask = cv2.resize(orig_mask.astype(np.uint8), (1024, 1024))
            orig_mask = orig_mask.astype(bool)
        # calculate the intersection over union
        intersection = np.logical_and(mask, orig_mask)
        union = np.logical_or(mask, orig_mask)
        iou = np.sum(intersection) / np.sum(union)
        iou_records[obj_id] = iou
    return iou_records if not return_vis else (iou_records, vises)

#%%
# parse the arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--support_image', type=str, help='Path to the support image.')
parser.add_argument('--support_mask', type=str, help='Path to the support segmentation mask.')
parser.add_argument('--trait_id', type=int, help='ID of the trait to retrieve, denoted by the integer value in the mask image.')
parser.add_argument('--query_images', type=str, help='Path to the query images folder.')
parser.add_argument('--output', type=str, help='Path to the output folder.')
parser.add_argument('--output_format', choices=["png", "gif"], default='gif', help='Output format (optional): gif, png.')
parser.add_argument('--top_k', type=int, default=5, help='Number of top-k retrieved images to save.')

args = parser.parse_args()
support_image_path = args.support_image
support_mask_path = args.support_mask
trait_id = args.trait_id
query_images_folder = args.query_images
output_folder = args.output
output_format = args.output_format
top_k = args.top_k

#%%
# load the support image and mask
print ("Loading support image and mask...")
support_image = cv2.imread(support_image_path)[..., ::-1]
support_mask = cv2.imread(support_mask_path, cv2.IMREAD_GRAYSCALE)
support_mask = (support_mask == trait_id).astype(np.uint8)

# load the query images
query_images = sorted(os.listdir(query_images_folder))
query_images = [cv2.imread(os.path.join(query_images_folder, img))[..., ::-1] for img in query_images]

# build the predictor
video_predictor = sam_utils.build_sam2_predictor(checkpoint="../checkpoints/sam2_hiera_large.pt")

retrieve_scores = []
retrieve_vis = []
print ("Retrieving images...")
for i, query_image in enumerate(tqdm(query_images)):
    iou_records, vises = cycle_consistency(support_image, query_image, [support_mask], video_predictor, return_vis=True)
    retrieve_scores.append(iou_records[0])
    retrieve_vis.append(vises[0])

# sort the query images based on the retrieval scores
sorted_indices = np.argsort(retrieve_scores)
sorted_retrieved_images = [retrieve_vis[i] for i in sorted_indices]

if output_format == "gif":
    # save the top-k retrieved images as a gif
    print (f"Saving the top-{top_k} retrieved images as a gif...")
    gif_images = [Image.fromarray(img) for img in sorted_retrieved_images[:top_k]]
    # resize the images to the same size
    min_shape = sorted(gif_images, key=lambda x: x.size)[0].size
    gif_images = [img.resize(min_shape, Image.ANTIALIAS) for img in gif_images]
    gif_images[0].save(os.path.join(output_folder, f"top_{top_k}_retrieved.gif"), save_all=True, append_images=gif_images[1:], duration=1000, loop=0)
else:
    # save the top-k retrieved images as png
    print (f"Saving the top-{top_k} retrieved images as png...")
    for i, img in enumerate(sorted_retrieved_images[:top_k]):
        cv2.imwrite(os.path.join(output_folder, f"retrieved_rank_{i+1}.png"), img[..., ::-1])

print("Done! Retrieval results saved at ", output_folder)