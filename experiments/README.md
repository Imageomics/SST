# Experiments

These are research scripts for OC-CCL (Open-Close Cycle Consistency Loss) finetuning and ablations. They are not part of the installable `sstrack` package and are excluded from the built distribution.

They predate the v2.0.0 transformers migration and still expect the vendored SAM2 copy (`sst.segment_anything_2`) and the local Hydra configs that lived under `src/sst/sam2_configs/`, both of which were removed when the package moved to the HuggingFace `transformers` backend. To run them as written, check out the pre-2.0.0 `v1.1.0` tag where that vendored code still exists.

Porting OC-CCL finetuning to the `transformers` `Sam2VideoModel` backend is tracked as a follow-up. `oc_ccl.py` and `butterfly_dataset.py` were moved here from `src/sst/` during packaging.
