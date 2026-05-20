# Input/Output Format Specifications

## Inputs

### Image files
- Format: JPEG or PNG (8-bit); PPM also supported
- For calibrated specimen images, use files with the `_calibrated` suffix

### Reference masks (`--support_mask`)
- Format: greyscale PNG, 8-bit
- Same spatial dimensions as the corresponding image
- Pixel values:

| Value | Wing |
|-------|------|
| 0 | Background |
| 1 | Right forewing |
| 2 | Left forewing |
| 3 | Right hindwing |
| 4 | Left hindwing |

### CSV file
- Columns: `image_path`, `group`, `mask_path`
- `mask_path` is empty for target frames
- Minimum 2 reference masks per group required

## Outputs

### Predicted masks
- Location: `<output_dir>/pred_masks/`
- Format: greyscale PNG, same pixel value convention as input masks
- One file per target image, named `<input_filename>.png`
