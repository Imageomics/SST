# Quick Reference: Running SST from Existing Masks

This guide assumes you already have reference masks. If you need to create them, see [Generating Reference Masks](reference-mask-tools/index.md).

## 1. Prepare your CSV

Create a CSV file that groups your images. Each row is one image; the group column controls which images are propagated together. SST requires a minimum of 2 reference masks per group.

```
image_path,group,mask_path
images/110_BOQ_D.jpg,BOQ,masks/110_BOQ_D_mask.png
images/111_BOQ_D.jpg,BOQ,masks/111_BOQ_D_mask.png
images/113_BOQ_D.jpg,BOQ,
```

Rows with a `mask_path` are treated as reference frames; rows without are targets.

## 2. Run inference

```bash
python inference.py --csv your_data.csv --output_dir output/
```

## 3. Check outputs

Results are written to `output/pred_masks/`. Each predicted mask is a greyscale PNG where pixel values correspond to wing IDs:

| Value | Wing |
|-------|------|
| 1 | Right forewing |
| 2 | Left forewing |
| 3 | Right hindwing |
| 4 | Left hindwing |
| 0 | Background |

## Notes

- Within-species grouping produces significantly better results than cross-species
- Reference mask quality is the upper bound on predicted mask quality — incomplete annotations propagate to all targets
- Reference frame selection uses `seed=0` internally, so it is deterministic but not user-controlled
