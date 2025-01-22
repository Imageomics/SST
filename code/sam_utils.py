import cv2
import groundingdino.util.inference as DINO_inf
import groundingdino.datasets.transforms as T
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
import sam2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse

def load_DINO_model(model_cfg_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_pretrained_path="checkpoints/groundingdino_swint_ogc.pth"):
    model = DINO_inf.load_model(model_cfg_path, model_pretrained_path)
    return model

def DINO_image_detection(img, text_prompt, box_conf=0.5, model=None, top_1=False):
    if model is None:
        GD_root = "/fs/scratch/PAS2099/danielf/SAM2/GroundingDINO"
        model_cfg_path = os.path.join(GD_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        model_pretrained_path = os.path.join(GD_root, "weights/groundingdino_swint_ogc.pth")
        model = DINO_inf.load_model(model_cfg_path, model_pretrained_path)
    TEXT_PROMPT = text_prompt
    BOX_TRESHOLD = box_conf
    TEXT_TRESHOLD = 0.25

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_pil = Image.fromarray(img)
    image_transformed, _ = transform(img_pil, None)

    boxes, logits, phrases = DINO_inf.predict(
        model=model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        remove_combined=True
    )
    h,w = img.shape[:2]
    boxes_cxcywh = boxes * torch.Tensor([w, h, w, h])
    boxes_cxcywh = boxes_cxcywh.numpy()
    boxes_xyxy = boxes_cxcywh.copy()
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    if top_1:
        if len(boxes_xyxy) == 0:
            return None
        return boxes_xyxy[np.argmax(logits)]
    return boxes_xyxy

def area(mask):
    if mask.size == 0: return 0
    return np.count_nonzero(mask) / mask.size

def show_mask(mask, ax, obj_id=None, random_color=False, borders = True, alpha=0.5):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, alpha])
    if not random_color and obj_id is not None:
        color = np.array([*plt.get_cmap("tab10")(obj_id)[:3], alpha])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def area(mask):
    if mask.size == 0: return 0
    return np.count_nonzero(mask) / mask.size

def nms_bbox_removal(boxes_xyxy, iou_thresh=0.25 ):
    remove_indices = []
    for i, box in enumerate(boxes_xyxy):
        for j in range(i+1, len(boxes_xyxy)):
            box2 = boxes_xyxy[j]
            iou1 = compute_iou(box, box2)
            iou2 = compute_iou(box2, box)
            if iou1 > iou_thresh or iou2 > iou_thresh:
                if iou1 > iou2:
                    remove_indices.append(j)
                else:
                    remove_indices.append(i)
    return [box for i, box in enumerate(boxes_xyxy) if i not in remove_indices]

def load_SAM2(ckpt_path, model_cfg_path):
    if torch.cuda.is_available():
        print("Using CUDA")
        device = "cuda"
    else:
        print("CUDA device not found, using CPU instead")
        device = "cpu"
    sam2 = build_sam2(model_cfg_path, ckpt_path, device=device, apply_postprocessing=False)
    return sam2

def compute_iou(box1, box2):
    # intersection / area of box1
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    if x5 >= x6 or y5 >= y6:
        return 0
    intersection = (x6 - x5) * (y6 - y5)
    union = (x2 - x1) * (y2 - y1)
    return intersection / union

def show_anns(anns, color=None, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].squeeze().shape[0], sorted_anns[0]['segmentation'].squeeze().shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation'].squeeze()
        if color is None:
            color_mask = np.concatenate([np.random.random(3), [0.75]])
        else:
            color_mask = color
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2) 

    ax.imshow(img)

def build_sam2_predictor(checkpoint="checkpoints/sam2_hiera_large.pt", model_cfg="sam2_hiera_l"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    return video_predictor

def load_masks(video_predictor, query_images, support_image, support_masks, offload_video_to_cpu=True, offload_state_to_cpu=True, verbose=False):
    '''
    video_predictor: sam2 predictor
    query_images: list of np.array of shape (H, W, 3)
    support_image: np.array of shape (H, W, 3)
    support_masks: list of np.array of shape (H, W)
    offload_video_to_cpu: for long video sequences, offload the video to the CPU to save GPU memory
    offload_state_to_cpu: save GPU memory by offloading the state to the CPU
    '''
    query_images.insert(0, support_image)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(None, image_inputs=query_images, async_loading_frames=False, offload_video_to_cpu=offload_video_to_cpu, offload_state_to_cpu=offload_state_to_cpu, verbose=verbose)
        video_predictor.reset_state(state)
        for i, patch_mask in enumerate(support_masks):
            ann_frame_idx = 0
            ann_obj_id = i  # give a unique id to each object we interact with
            patch_mask = np.array(patch_mask, dtype=np.uint8)
            patch_mask = cv2.resize(patch_mask, (1024, 1024))
            _, _, _ = video_predictor.add_new_mask(
                inference_state=state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=patch_mask,
            )
    return state

def propagate_masks(video_predictor, state, verbose=False):
    """
    returns: list[dict] with keys 'obj_ids', 'segmentation', 'area'
    list['segmentation']: np.array of shape (H, W) with dtype bool
    """
    frame_info = []
    # run propagation throughout the video and collect the results in a dict
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state, verbose=verbose):
            out_mask_logits = (out_mask_logits>0).cpu().numpy().squeeze()
            if out_mask_logits.ndim == 2:
                out_mask_logits = np.expand_dims(out_mask_logits, axis=0)
            frame_info.append({'obj_ids': out_obj_ids, 'segmentation': out_mask_logits, 'area': area(out_mask_logits)})
    return frame_info

def show_video_masks(image, frame_info):
    img_resized = cv2.resize(image, (1024, 1024))
    plt.imshow(img_resized)
    for obj_ids, mask in zip(frame_info['obj_ids'], frame_info['masks']):
        mask = cv2.resize(mask.astype(np.uint8), (1024, 1024))
        show_mask(mask, plt.gca(), obj_id=obj_ids, borders=True, alpha=0.75)
    plt.axis('off')
    plt.show()

def get_parser(inputs):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args(inputs)
    return args

def auto_segment_SAM(boxes_xyxy, img, iou_thresh=0.9, stability_score_thresh=0.95, min_mask_region_area=10000, verbose=False):
    checkpoint = "../../checkpoints/sam2_hiera_large.pt"
    model_cfg = "../../sam2_configs/sam2_hiera_l.yaml"
    sam2 = load_SAM2(checkpoint, model_cfg)
    auto_mask_predictor = SAM2AutomaticMaskGenerator(sam2, 
                                                    points_per_batch=128, 
                                                    pred_iou_thresh=iou_thresh,
                                                    stability_score_thresh=stability_score_thresh, 
                                                    min_mask_region_area=min_mask_region_area, 
                                                    multimask_output=True)
    masks_list = []
    for box_xyxy in boxes_xyxy:
        wing = img[int(box_xyxy[1]):int(box_xyxy[3]), int(box_xyxy[0]):int(box_xyxy[2])]
        mask = auto_mask_predictor.generate(wing)
        # for mask_
        # dict in mask:
        #     mask_dict['segmentation'] = np.bitwise_not(mask_dict['segmentation'])
        if verbose:
            plt.imshow(wing)
            show_anns(mask)
            # remove axis
            plt.axis('off')
            plt.show()
        # translate the mask to the original image
        binary_masks = [e['segmentation'] for e in mask]

        for e in binary_masks:
            new_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            new_mask[int(box_xyxy[1]):int(box_xyxy[3]), int(box_xyxy[0]):int(box_xyxy[2])] = e
            new_mask_dict = {
                'segmentation': new_mask,
                'area': area(new_mask)
            }
            masks_list.append(new_mask_dict)
    return masks_list

def show_masks(masks_list, img, verbose=True, imshow=True, grey=False):
    if imshow:
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
    plt.axis('off')
    show_anns(masks_list)
    if verbose:
        plt.show()

def show_individual_masks(masks_list, img):
    for mask in masks_list:
        plt.imshow(img)
        plt.axis('off')
        show_anns([mask])
        plt.show()