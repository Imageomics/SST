"""Tests for the mask preparation utilities that depend only on PIL, numpy, and cv2."""

import argparse

import numpy as np
from PIL import Image

from sst import get_mask_from_crop, prepare_starter_mask


def test_prepare_starter_mask_maps_white_to_object_id(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:5, 2:5] = 255
    in_path = tmp_path / "mask.png"
    out_path = tmp_path / "out.png"
    Image.fromarray(arr).save(in_path)

    args = argparse.Namespace(mask_image_path=str(in_path), mask_image_path_out=str(out_path))
    prepare_starter_mask.run(args)

    result = np.array(Image.open(out_path))
    assert set(np.unique(result).tolist()) == {0, 5}
    assert (result[2:5, 2:5] == 5).all()


def test_get_mask_from_crop_locates_patch(tmp_path):
    # A 30x30 source with a distinctive 8x8 patch the crop is taken from.
    source = np.zeros((30, 30, 3), dtype=np.uint8)
    source[10:18, 12:20] = [200, 50, 25]
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)

    crop = np.zeros((8, 8, 4), dtype=np.uint8)
    crop[..., :3] = source[10:18, 12:20]
    crop[..., 3] = 255
    crop_path = tmp_path / "crop.png"
    Image.fromarray(crop, mode="RGBA").save(crop_path)

    out_path = tmp_path / "mask.png"
    args = argparse.Namespace(
        image_path=str(source_path), image_crop_path=str(crop_path), mask_image_path_out=str(out_path)
    )
    get_mask_from_crop.run(args)

    mask = np.array(Image.open(out_path))
    assert mask.shape == (30, 30)
    assert (mask[10:18, 12:20] == 255).all()
    assert mask.sum() == 255 * 64
