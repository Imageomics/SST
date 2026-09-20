# Generating Reference Masks

SST requires at least 2 manually annotated reference masks per group before it can run inference. This section documents four tools you can use to create them, but any annotation tool may be used. The four tools listed here are a representative selection and the choice is arbitrary -- what matters is producing the correct output format.

## What SST expects as input

Regardless of which annotation tool you use, every reference mask must be a greyscale PNG image with the same spatial dimensions as the corresponding butterfly image. Pixel values encode wing identity as integers:

| Value | Wing |
|-------|------|
| 0 | Background |
| 1 | Right forewing |
| 2 | Left forewing |
| 3 | Right hindwing |
| 4 | Left hindwing |

This is the format SST's `--support_mask` argument expects. Any annotation tool that can produce or be converted to this format is compatible with SST.

## Getting a sample dataset

The examples in this section use the [Heliconius Collection (Cambridge Butterfly)](https://huggingface.co/datasets/imageomics/Heliconius-Collection_Cambridge-Butterfly) dataset, which contains over 36,000 dorsal and ventral images of Heliconius specimens. The following snippet retrieves 20 dorsal images using the HuggingFace Hub:

```python
from datasets import load_dataset
import os

dataset = load_dataset(
    "imageomics/Heliconius-Collection_Cambridge-Butterfly",
    "dorsal",
    split="train"
)

# Take a small subsample of 20 images
sample = dataset.select(range(20))

# Save images to a local directory
os.makedirs("heliconius_sample", exist_ok=True)
for i, example in enumerate(sample):
    example["image"].save(f"heliconius_sample/{i:02d}.jpg")

print(f"Saved {len(sample)} images to heliconius_sample/")
```

Install the required packages first:

```bash
pip install datasets pillow
```

## Available annotation tools

| Tool | Approach | Export format |
|------|----------|---------------|
| [napari](napari.md) | Python script with interactive Labels layer | PNG (direct) |
| [X-AnyLabeling](x-anylabeling.md) | GUI polygon annotation | JSON converted to PNG |
| [Fiji](fiji.md) | ROI Manager + macro | PNG (direct via macro) |
| [Label Studio](label-studio.md) | Web UI with optional SAM2 backend | Brush annotation exported as PNG |
