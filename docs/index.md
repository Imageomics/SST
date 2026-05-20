# SST — Static Segmentation by Tracking

## Frustratingly Label-Efficient Segmentation for Biological Specimens

[![GitHub](https://img.shields.io/badge/GitHub-Imageomics%2FSST-blue)](https://github.com/Imageomics/SST)

SST is a pipeline for propagating reference segmentation masks across a set of images using SAM2’s video tracking capabilities. It was designed for biological image analysis workflows where a small number of manually annotated reference masks can be propagated to a large set of target images.

## Package Purpose

SST addresses the challenge of segmenting recurring structures — such as butterfly wings — across large collections of specimen images. Rather than annotating every image individually, a researcher annotates a small number of reference frames and SST propagates those masks to all remaining images in the group.

The pipeline was developed as part of the [LepidopteraLens](https://github.com/Imageomics/LepidopteraLens) workflow for automated trait extraction from pinned Lepidoptera specimens.

## Key Concepts

- **Reference frames** — images for which you provide manually annotated masks
- **Target frames** — images SST propagates masks to automatically
- **Grouping** — images are grouped (e.g. by species) so propagation happens within visually similar sets; within-species grouping produces significantly better results than cross-species

## Limitations

- Reference mask quality is the upper bound on predicted mask quality — incomplete annotations propagate to all targets
- SAM2 automatic mask generation is the bottleneck for multi-specimen images
- Reference frame selection uses a fixed random seed internally and is not user-controlled

## Getting Started

To get started with SST, see the [Quick Reference](quickstart.md) guide.

---

> **Note:** Documentation should only be published after PyPI packaging is complete. See [issue #22](https://github.com/Imageomics/SST/issues/22).
