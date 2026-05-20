# Generating a Reference Mask with napari

`napari_annotate.py` opens a butterfly specimen image as a napari Labels layer and lets you trace each wing using the polygon tool, assigning integers 1–4 for right forewing, left forewing, right hindwing, and left hindwing — matching the YOLO class convention. A docked Save button casts the Labels array from `int32` to `uint8` and writes it as a greyscale PNG via `imageio.imwrite`, producing exactly the format SST's `--support_mask` expects. The `PushButton` plus `add_dock_widget` pattern also makes this script a direct structural prototype for the future `napari-sst` plugin.

## Installation

```bash
pip install "napari[all]==0.7.0" imageio scikit-image magicgui
```

## Script

Save the following as `napari_annotate.py` in your SST directory:

```python
import numpy as np
import imageio.v3 as iio
from skimage import io
import napari
from magicgui.widgets import PushButton

IMAGE_PATH = "your_image.jpg"   # change this
OUTPUT_PATH = "mask.png"        # change this

image = io.imread(IMAGE_PATH)
viewer = napari.Viewer()
viewer.add_image(image, name="butterfly")
labels_layer = viewer.add_labels(
    np.zeros(image.shape[:2], dtype=np.int32),
    name="wings"
)

def save_mask():
    mask = labels_layer.data.astype(np.uint8)
    iio.imwrite(OUTPUT_PATH, mask)
    print(f"Saved mask to {OUTPUT_PATH}")

btn = PushButton(label="Save Mask")
btn.clicked.connect(save_mask)
viewer.window.add_dock_widget(btn, area="right")
napari.run()
```

## Usage

1. Edit `IMAGE_PATH` and `OUTPUT_PATH` at the top of the script
2. Run: `python napari_annotate.py`
3. In the viewer, select the **Labels** layer
4. Choose the **polygon** tool from the toolbar
5. Set the label value to the wing you are tracing:
   - `1` = right forewing
   - `2` = left forewing
   - `3` = right hindwing
   - `4` = left hindwing
6. Trace each wing, then click **Save Mask** in the right dock panel

## Notes

- The Save button writes a uint8 PNG with values 0–4, directly usable as SST's `--support_mask`
- Within-species propagation (e.g. CAM→CAM) produces clean results; cross-species propagation shows expected degradation consistent with SST's general behaviour
