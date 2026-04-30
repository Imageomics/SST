"""
ButterflyOCCCLDataset — Dataset for One-shot Cycle-Consistency Learning.

Returns (x0, y0, xu) triplets from the same species:
  x0: reference image (has ground-truth mask y0)
  y0: ground-truth mask for x0
  xu: unlabeled query image from the same species

Images loaded from data/cambridge_butterfly/images/
Masks loaded from DataSet_Butterfly/*/pkl/
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "cambridge_butterfly"
SPLIT_DIR = DATA_ROOT / "train_test_separate"
IMAGE_DIR = DATA_ROOT / "images"


class ButterflyOCCCLDataset(Dataset):
    """
    One-shot cycle-consistency dataset for butterfly segmentation.

    Each sample returns a (reference, mask, query) triplet from the same species.
    The reference and query are different images.
    """

    def __init__(
        self,
        species=None,
        split="train",
        image_size=1024,
        data_root=None,
    ):
        """
        Args:
            species: list of species names to include (None = all 5 major species)
            split: "train" or "test"
            image_size: resize images to this size (square)
            data_root: override default data root path
        """
        self.image_size = image_size
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.image_dir = self.data_root / "images"
        self.split_dir = self.data_root / "train_test_separate"

        if species is None:
            species = [d.name for d in sorted(self.split_dir.iterdir()) if d.is_dir()]

        # Load entries per species
        self.species_entries = {}  # species_name -> list of (image_id, url, mask_path)
        self.all_entries = []  # flat list of (species_name, image_id, url, mask_path)

        for sp in species:
            sp_dir = self.split_dir / sp
            json_file = sp_dir / f"{split}_data.json"
            if not json_file.exists():
                continue
            with open(json_file) as f:
                data = json.load(f)

            # Filter to entries that have both image and mask available
            valid = []
            for entry in data:
                image_id, url, mask_rel = entry[0], entry[1], entry[2]
                mask_path = self.data_root / mask_rel
                # Check image exists (try common extensions)
                img_path = self._find_image(image_id, url)
                if img_path is not None and mask_path.exists():
                    valid.append((image_id, url, str(mask_path), str(img_path)))

            if len(valid) >= 2:  # need at least 2 images for pairs
                self.species_entries[sp] = valid
                for v in valid:
                    self.all_entries.append((sp, *v))

        print(f"ButterflyOCCCLDataset: {len(self.all_entries)} samples across "
              f"{len(self.species_entries)} species")

    def _find_image(self, image_id, url):
        """Find the downloaded image file for a given image_id."""
        ext = Path(url).suffix  # .JPG, .CR2, etc.
        img_path = self.image_dir / f"{image_id}{ext}"
        if img_path.exists():
            return img_path
        return None

    def __len__(self):
        return len(self.all_entries)

    def _load_image(self, path):
        """Load and resize an image to (image_size, image_size, 3) RGB."""
        path = str(path)
        if path.lower().endswith(('.cr2', '.nef', '.arw', '.dng')):
            import rawpy
            with rawpy.imread(path) as raw:
                img = raw.postprocess()
        else:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    def _load_mask(self, path):
        """Load and resize a mask to (image_size, image_size) binary."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        species, image_id, url, mask_path, img_path = self.all_entries[idx]

        # x0: reference image, y0: its mask
        x0 = self._load_image(img_path)
        y0 = self._load_mask(mask_path)

        # xu: query image from same species (different from reference)
        candidates = self.species_entries[species]
        query = random.choice(candidates)
        while query[0] == image_id and len(candidates) > 1:
            query = random.choice(candidates)
        xu = self._load_image(query[3])

        # Convert to tensors: images as (3, H, W) float32 [0, 1], mask as (1, H, W)
        x0 = torch.from_numpy(x0).permute(2, 0, 1).float() / 255.0
        xu = torch.from_numpy(xu).permute(2, 0, 1).float() / 255.0
        y0 = torch.from_numpy(y0).unsqueeze(0)

        return x0, y0, xu
