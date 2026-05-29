"""Tests for the pure numpy and cv2 helpers in sam_utils that do not require torch."""

import numpy as np

from sst import sam_utils


def test_area_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True
    assert sam_utils.area(mask) == 0.5
    assert sam_utils.area(np.array([])) == 0


def test_compute_iou_overlap_and_disjoint():
    # Identical boxes: intersection equals box1 area, so the ratio is 1.
    assert sam_utils.compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    # Disjoint boxes have no intersection.
    assert sam_utils.compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    # Half overlap of box1.
    assert sam_utils.compute_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5


def test_nms_bbox_removal_drops_overlaps():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (100, 100, 110, 110)]
    kept = sam_utils.nms_bbox_removal(boxes, iou_thresh=0.25)
    assert (100, 100, 110, 110) in kept
    assert len(kept) == 2
