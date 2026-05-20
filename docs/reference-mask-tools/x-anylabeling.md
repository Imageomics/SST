# Generating a Reference Mask with X-AnyLabeling

X-AnyLabeling is a GUI annotation tool that supports polygon labeling and exports annotations as JSON. A short Python conversion script is needed to produce the uint8 PNG format SST expects.

## Installation

X-AnyLabeling pins `onnxruntime==1.16.0`, which has no Python 3.12 wheels. Use the standalone macOS arm64 app instead:

1. Go to [https://github.com/CVHub520/X-AnyLabeling/releases](https://github.com/CVHub520/X-AnyLabeling/releases)
2. Download the latest `x-anylabeling-macos-arm64` release
3. Open the `.dmg`, drag to Applications, and launch

## Usage

1. Launch X-AnyLabeling from Applications
2. Open your butterfly image via **File → Open Image**
3. Select the **Polygon** tool from the left toolbar
4. Trace each wing and assign a label name when prompted:
   - `right_forewing`
   - `left_forewing`
   - `right_hindwing`
   - `left_hindwing`
5. Annotations are auto-saved as a JSON file alongside the image

> **Note:** The built-in MASK export button throws a type error with polygon annotations in the current release. Use the conversion script below instead.

## Converting JSON to mask PNG

After annotating, run this script to convert the JSON to a uint8 greyscale PNG:

```python
import json
import numpy as np
from PIL import Image, ImageDraw

JSON_PATH = "52_SAG_D.json"      # change this
OUTPUT_PATH = "mask.png"         # change this

LABEL_MAP = {
    "right_forewing": 1,
    "left_forewing": 2,
    "right_hindwing": 3,
    "left_hindwing": 4,
}

with open(JSON_PATH) as f:
    data = json.load(f)

h, w = data["imageHeight"], data["imageWidth"]
mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
draw = ImageDraw.Draw(mask)

for shape in data["shapes"]:
    label = shape["label"]
    val = LABEL_MAP.get(label, 0)
    if val == 0:
        continue
    pts = [(int(x), int(y)) for x, y in shape["points"]]
    draw.polygon(pts, fill=val)

mask.save(OUTPUT_PATH)
print("shape:", np.array(mask).shape, "| labels:", np.unique(np.array(mask)))
```

Expected output:
```
shape: (2499, 2939) | labels: [0 1 2 3 4]
```

## Notes

- The JSON format follows the standard LabelMe schema with a `shapes` key containing polygon point lists
- The output is a uint8 PNG with values 0–4, directly usable as SST's `--support_mask`
- SST results validated clean across all seven calibrated species using this workflow
