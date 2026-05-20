# Generating a Reference Mask with Fiji

Fiji is a distribution of ImageJ with plugins pre-installed. It uses the ROI Manager to collect polygon selections for each wing, then a macro fills them into a blank 8-bit image and saves it as PNG. No Python script is needed.

## Installation

1. Go to [https://imagej.net/software/fiji/downloads](https://imagej.net/software/fiji/downloads)
2. Download the **macOS arm64** version
3. Unzip and drag Fiji to your Applications folder
4. Launch Fiji from Applications

## Usage

### 1. Open and prepare the image

1. Open your butterfly image: **File → Open**
2. Convert to 8-bit: **Image → Type → 8-bit**

### 2. Trace wings with the ROI Manager

1. Open the ROI Manager: **Analyze → Tools → ROI Manager**
2. Select the **Polygon** tool from the Fiji toolbar
3. Trace the right forewing, then press **t** to add it to the ROI Manager
4. Double-click the ROI in the manager and rename it `right_forewing`
5. Repeat for each wing:
   - `left_forewing`
   - `right_hindwing`
   - `left_hindwing`

### 3. Run the export macro

1. Open the macro editor: **Plugins → Macros → Edit**
2. Paste the following macro, updating the output path:

```
names = newArray("right_forewing", "left_forewing", "right_hindwing", "left_hindwing");
values = newArray(1, 2, 3, 4);

w = getWidth();
h = getHeight();
newImage("mask", "8-bit black", w, h, 1);

for (i = 0; i < names.length; i++) {
    idx = -1;
    for (j = 0; j < roiManager("count"); j++) {
        roiManager("select", j);
        if (Roi.getName() == names[i]) {
            idx = j;
            j = roiManager("count");
        }
    }
    if (idx >= 0) {
        roiManager("select", idx);
        setForegroundColor(values[i], values[i], values[i]);
        fill();
    }
}

saveAs("PNG", "/Users/sahasra/SST/mask.png");
```

3. Update `/Users/sahasra/SST/mask.png` to your desired output path
4. Click **Run**

## Notes

- The output is a uint8 PNG with values 0–4, directly usable as SST's `--support_mask`
- No Python conversion step is needed — the macro handles everything natively
- SST results validated clean across all seven calibrated species using this workflow
