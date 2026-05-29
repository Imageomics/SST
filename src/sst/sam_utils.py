"""Model utilities for Static Segmentation by Tracking.

This module wraps the HuggingFace ``transformers`` implementations of SAM2 and Grounding DINO so the
rest of the package can stay backend agnostic. SAM2 video tracking is driven through the
``Sam2Tracker`` wrapper, text prompted detection through ``detect_text``, and automatic mask
generation through ``auto_segment``. Model weights are fetched from the HuggingFace Hub on first use
and cached locally, so no manual checkpoint download is required.

Object identifiers follow the same zero based convention used throughout the package: a support mask
image with pixel values ``1..N`` is decomposed into ``N`` binary masks whose object ids are ``0..N-1``,
and the tracker returns per object masks in that same order.
"""

import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np

# torch and transformers are imported lazily inside the functions and classes that need them. This
# keeps the pure numpy and cv2 helpers (visualization, IoU, NMS) importable without paying the cost
# of loading the deep learning stack.

DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"


def resolve_device(device=None):
    """Return a concrete torch device, defaulting to CUDA when available and CPU otherwise."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device):
    """Pick a compute dtype for the device. CUDA uses bfloat16; CPU stays in float32.

    Running bfloat16 on CPU is both unsupported by several ops and far slower, so CPU installs must
    use float32 to be usable at all.
    """
    import torch

    device = torch.device(device)
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _to_pil(image):
    """Convert an RGB numpy array or path-like to a PIL image, leaving PIL images untouched."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return Image.open(image).convert("RGB")


class Sam2Tracker:
    """SAM2 video predictor wrapper that owns the model, processor, device, and dtype.

    The tracker treats the support image followed by the query images as the frames of a short video.
    Reference masks are attached to the first frame and propagated across the remaining frames. This
    replaces the previous ``build_sam2_predictor`` / ``load_masks`` / ``propagate_masks`` trio, which
    assumed the local SAM2 checkpoint API rather than the ``transformers`` inference session API.
    """

    def __init__(self, model_id=DEFAULT_SAM2_MODEL, device=None, dtype=None):
        import transformers

        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype if dtype is not None else resolve_dtype(self.device)
        self.model = transformers.Sam2VideoModel.from_pretrained(model_id).to(
            self.device, dtype=self.dtype
        )
        self.processor = transformers.Sam2VideoProcessor.from_pretrained(model_id)

    def segment(self, support_image, support_masks, query_images, verbose=False):
        """Propagate reference masks from the support frame across the query frames.

        ``support_image`` is an RGB array, ``support_masks`` a list of boolean arrays (one per object,
        in object id order), and ``query_images`` a list of RGB arrays. Returns one frame info dict per
        frame in ``[support_image, *query_images]`` order, each with keys ``obj_ids`` (the zero based
        ids), ``segmentation`` (a ``(num_objects, H, W)`` boolean array at the support image size), and
        ``area`` (mask coverage fraction).
        """
        import torch

        support_pil = _to_pil(support_image)
        target_wh = support_pil.size  # (width, height)
        target_h, target_w = target_wh[1], target_wh[0]

        frames = [support_pil] + [_to_pil(q).resize(target_wh) for q in query_images]

        num_objects = len(support_masks)
        input_masks = []
        for mask in support_masks:
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST)
            input_masks.append(torch.from_numpy(mask.astype(np.float32)))

        session = self.processor.init_video_session(
            video=frames, inference_device=self.device, dtype=self.dtype
        )
        # add_inputs_to_inference_session consumes the obj_ids list it is given, so pass a throwaway
        # copy and keep num_objects as the source of truth for output shapes.
        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(range(num_objects)),
            input_masks=input_masks,
        )

        frame_info = [None] * len(frames)
        for output in self.model.propagate_in_video_iterator(
            session, start_frame_idx=0, show_progress_bar=verbose
        ):
            masks = self.processor.post_process_masks(
                [output.pred_masks], original_sizes=[[target_h, target_w]], binarize=True
            )[0]
            masks = masks[:, 0, :, :].cpu().numpy().astype(bool)

            # Map each output row to its object id. SAM2 may omit objects it considers absent from a
            # frame, so rely on the per-frame object_ids rather than assuming all objects are present.
            segmentation = np.zeros((num_objects, target_h, target_w), dtype=bool)
            for row, obj_id in enumerate(output.object_ids):
                segmentation[obj_id] = masks[row]

            frame_info[output.frame_idx] = {
                "obj_ids": list(range(num_objects)),
                "segmentation": segmentation,
                "area": area(segmentation),
            }
        return frame_info


@functools.lru_cache(maxsize=2)
def _load_dino(model_id, device):
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model


def detect_text(
    image,
    text_prompt,
    box_threshold=0.4,
    text_threshold=0.25,
    model_id=DEFAULT_DINO_MODEL,
    device=None,
    top_1=False,
):
    """Detect objects matching a text prompt and return their bounding boxes in xyxy pixel format.

    Uses Grounding DINO via ``transformers``. Grounding DINO expects lowercase prompts with classes
    separated by periods, for example ``"a wing. an antenna."``. Returns an ``(N, 4)`` array of xyxy
    boxes, or the single highest scoring box when ``top_1`` is set (or ``None`` if nothing is found).
    """
    import torch

    device = resolve_device(device)
    processor, model = _load_dino(model_id, str(device))

    pil = _to_pil(image)
    inputs = processor(images=pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(pil.height, pil.width)],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    if top_1:
        if len(boxes) == 0:
            return None
        scores = results["scores"].cpu().numpy()
        return boxes[int(np.argmax(scores))]
    return boxes


def auto_segment(boxes_xyxy, img, model_id="facebook/sam2.1-hiera-large", device=None,
                 points_per_batch=128, pred_iou_thresh=0.9, stability_score_thresh=0.95,
                 min_mask_region_area=10000, verbose=False):
    """Generate masks automatically within each box using the SAM2 mask generation pipeline.

    This is the automatic (non tracked) counterpart to ``Sam2Tracker``. It crops the image to each
    box, runs the HuggingFace ``mask-generation`` pipeline, and translates the resulting masks back
    into full image coordinates. Returns a list of dicts with ``segmentation`` (full image boolean
    array) and ``area``.
    """
    import transformers

    device = resolve_device(device)
    generator = transformers.pipeline(
        "mask-generation", model=model_id, device=device, points_per_batch=points_per_batch
    )

    masks_list = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        crop = img[y0:y1, x0:x1]
        outputs = generator(
            _to_pil(crop),
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        for crop_mask in outputs["masks"]:
            crop_mask = np.asarray(crop_mask, dtype=bool)
            full = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            full[y0:y1, x0:x1] = crop_mask
            if area(full) * full.size < min_mask_region_area:
                continue
            masks_list.append({"segmentation": full, "area": area(full)})
        if verbose:
            plt.imshow(crop)
            plt.axis("off")
            plt.show()
    return masks_list


def area(mask):
    """Fraction of the array that is set, returning 0 for empty arrays."""
    if mask.size == 0:
        return 0
    return np.count_nonzero(mask) / mask.size


def compute_iou(box1, box2):
    """Intersection of two xyxy boxes divided by the area of the first box."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    if x5 >= x6 or y5 >= y6:
        return 0
    intersection = (x6 - x5) * (y6 - y5)
    union = (x2 - x1) * (y2 - y1)
    return intersection / union


def nms_bbox_removal(boxes_xyxy, iou_thresh=0.25):
    """Drop overlapping boxes, keeping the one with the larger relative overlap in each conflict."""
    remove_indices = []
    for i, box in enumerate(boxes_xyxy):
        for j in range(i + 1, len(boxes_xyxy)):
            box2 = boxes_xyxy[j]
            iou1 = compute_iou(box, box2)
            iou2 = compute_iou(box2, box)
            if iou1 > iou_thresh or iou2 > iou_thresh:
                remove_indices.append(j if iou1 > iou2 else i)
    return [box for i, box in enumerate(boxes_xyxy) if i not in remove_indices]


def show_mask(mask, ax, obj_id=None, random_color=False, borders=True, alpha=0.5):
    """Overlay a single binary mask on a matplotlib axis, optionally with smoothed contours."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    if not random_color and obj_id is not None:
        color = np.array([*plt.get_cmap("tab10")(obj_id)[:3], alpha])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_anns(anns, color=None, borders=True):
    """Overlay a list of automatic mask annotations (dicts with a ``segmentation`` key)."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    first = sorted_anns[0]["segmentation"].squeeze()
    img = np.ones((first.shape[0], first.shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"].squeeze()
        color_mask = np.concatenate([np.random.random(3), [0.75]]) if color is None else color
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2)
    ax.imshow(img)


def show_masks(masks_list, img, verbose=True, imshow=True, grey=False):
    """Display an image with all automatic masks overlaid."""
    if imshow:
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
    plt.axis("off")
    show_anns(masks_list)
    if verbose:
        plt.show()


def show_individual_masks(masks_list, img):
    """Display each automatic mask on its own figure."""
    for mask in masks_list:
        plt.imshow(img)
        plt.axis("off")
        show_anns([mask])
        plt.show()
