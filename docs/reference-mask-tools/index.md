# Generating Reference Masks

SST requires at least 2 manually annotated reference masks per group before it can run inference. This section documents four tools you can use to create them.

All tools must produce a greyscale PNG where pixel values encode wing identity:

| Value | Wing |
|-------|------|
| 0 | Background |
| 1 | Right forewing |
| 2 | Left forewing |
| 3 | Right hindwing |
| 4 | Left hindwing |

| Tool | Approach | Export format |
|------|----------|---------------|
| [napari](napari.md) | Python script with interactive Labels layer | PNG (direct) |
| [X-AnyLabeling](x-anylabeling.md) | GUI polygon annotation | JSON converted to PNG |
| [Fiji](fiji.md) | ROI Manager + macro | PNG (direct via macro) |
| [Label Studio](label-studio.md) | Web UI with optional SAM2 backend | Brush annotation exported as PNG |
