# Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation
[Imageomics Institute](https://imageomics.osu.edu/)

[Zhenyang Feng](https://defisch.github.io/), Zihe Wang, Saul Ibaven Bueno, Tomasz Frelek, Advikaa Ramesh, Jingyan Bai, Lemeng Wang, [Zanming Huang](https://tzmhuang.github.io/), [Jianyang Gu](https://vimar-gu.github.io/), [Jinsu Yoo](https://jinsuyoo.info/), [Tai-Yu Pan](https://tydpan.github.io/), Arpita Chowdhury, Michelle Ramirez, [Elizabeth G Campolongo](https://u.osu.edu/campolongo-4/), Matthew J Thompson, [Christopher G. Lawrence](https://eeb.princeton.edu/people/christopher-lawrence), [Sydne Record](https://umaine.edu/wle/faculty-staff-directory/sydne-record/), [Neil Rosser](https://people.miami.edu/profile/74f02be76bd3ae57ed9edfdad0a3f76d), [Anuj Karpatne](https://anujkarpatne.github.io/), [Daniel Rubenstein](https://eeb.princeton.edu/people/daniel-rubenstein), [Hilmar Lapp](https://lappland.io/), [Charles V. Stewart](https://www.cs.rpi.edu/~stewart/), Tanya Berger-Wolf, [Yu Su](https://ysu1989.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao)

[[arXiv]](https://arxiv.org/abs/2501.06749) [[Dataset]](https://github.com/Imageomics/NEON_beetles_masks.git) [[BibTeX]](#-citation)

![main figure](assets/main.png)

## 🗓️ TODO
- [x] Release inference code
- [x] Release beetle part segmentation dataset
- [ ] Release online demo
- [ ] Release one-shot fine-tuning (OC-CCL) code
- [ ] Release trait retrieval code
- [ ] Release butterfly trait segmentation dataset

## 🛠️ Installation
To use SST, the following setup must be ran on a GPU enabled machine. The code requires `torch>=2.5.0`, and `python=3.10.14` is recommended.

Example Conda Environment Setup:
```bash
# clone repo
(git clone https://github.com/Imageomics/SST.git && cd SST)
# Create conda environment
conda create --name sst python=3.10.14
conda activate sst
# Download PyTorch corresponding to the CUDA version of the GPU
...
# Download and setup GroundingDINO
(git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO/ && pip install -e .)
# Download required python packages
pip install -r requirements.txt --no-dependencies
# Download model checkpoints
(cd checkpoints && ./download_ckpts.sh)
(cd checkpoints && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
# Install SAM 2
(cd segment-anything-2 && pip install -e .)
```

## 🧑‍💻 Usage


```bash
python code/segment.py --support_image /path/to/sample/image.png \
  --support_mask /path/to/greyscale_mask.png \ 
  --query_images /path/to/query/images/folder \
  --output /path/to/output/folder \
  [--output_format png/gif]
```

## 📊 Dataset
Beetle part segmentation dataset is out! Available [here](https://github.com/Imageomics/NEON_beetles_masks.git).
We will release our trait segmentation datasets for butterfly in the near future!

## ❤️ Acknowledgements
This project makes use of the [SAM2](https://github.com/facebookresearch/sam2) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) codebases. We are grateful to the developers and maintainers of these projects for their contributions to the open-source community.
We thank [LoRA](https://github.com/microsoft/LoRA) for their great work.


## 📝 Citation
If you find our work helpful for your research, please consider citing using the following BibTeX entry:
```bibtex
@article{feng2025static,
  title={Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation},
  author={Feng, Zhenyang and Wang, Zihe and Bueno, Saul Ibaven and Frelek, Tomasz and Ramesh, Advikaa and Bai, Jingyan and Wang, Lemeng and Huang, Zanming and Gu, Jianyang and Yoo, Jinsu and others},
  journal={arXiv preprint arXiv:2501.06749},
  year={2025}
}
```
