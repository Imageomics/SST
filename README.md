# Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation
[Imageomics Institute](https://imageomics.osu.edu/)

[Zhenyang Feng](https://defisch.github.io/), [Zihe Wang](https://ziheherzwang.github.io/HerzWangWebsite/), Saul Ibaven Bueno, Tomasz Frelek, Advikaa Ramesh, Jingyan Bai, Lemeng Wang, [Zanming Huang](https://tzmhuang.github.io/), [Jianyang Gu](https://vimar-gu.github.io/), [Jinsu Yoo](https://jinsuyoo.info/), [Tai-Yu Pan](https://tydpan.github.io/), Arpita Chowdhury, Michelle Ramirez, [Elizabeth G Campolongo](https://u.osu.edu/campolongo-4/), Matthew J Thompson, [Christopher G. Lawrence](https://eeb.princeton.edu/people/christopher-lawrence), [Sydne Record](https://umaine.edu/wle/faculty-staff-directory/sydne-record/), [Neil Rosser](https://people.miami.edu/profile/74f02be76bd3ae57ed9edfdad0a3f76d), [Anuj Karpatne](https://anujkarpatne.github.io/), [Daniel Rubenstein](https://eeb.princeton.edu/people/daniel-rubenstein), [Hilmar Lapp](https://lappland.io/), [Charles V. Stewart](https://www.cs.rpi.edu/~stewart/), Tanya Berger-Wolf, [Yu Su](https://ysu1989.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao)

[[arXiv]](https://arxiv.org/abs/2501.06749) [[Dataset]](https://github.com/Imageomics/NEON_beetles_masks.git) [[BibTeX]](#-citation)

![main figure](assets/main.png)

## 🗓️ TODO
- [x] Release inference code
- [x] Release beetle part segmentation dataset
- [ ] Release online demo
- [ ] Release one-shot fine-tuning (OC-CCL) code
- [x] Release trait retrieval code
- [x] Release butterfly trait segmentation dataset

## 🛠️ Installation
To use SST, the following setup must be ran on a GPU enabled machine. The code requires `torch>=2.5.0`, and `python=3.10.14` is recommended. (Note: Make sure your system is using GXX and GCC compilers)

Example Conda Environment Setup:
```bash
# Clone repo
git clone https://github.com/Imageomics/SST.git
cd SST
# Create conda environment
conda create --name sst python=3.10.14
conda activate sst
# Download the version of PyTorch that matches the GPU's CUDA version, see https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url ...
# Download and setup GroundingDINO
(git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO/ && pip install -e .)
# Install SAM 2
(cd segment-anything-2 && pip install -e .)
# Download required python packages, ignore the conflicts message
pip install -r requirements.txt --no-dependencies
# Download model checkpoints
(cd checkpoints && ./download_ckpts.sh)
(cd checkpoints && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
```

## 🧑‍💻 Usage

### Segmentation
For one-shot trait/part segmentation, please run the following demo code:
```bash
python code/segment.py --support_image /path/to/sample/image.png \
  --support_mask /path/to/greyscale_mask.png \ 
  --query_images /path/to/query/images/folder \
  --output /path/to/output/folder \
  --output_format "png" # png or gif, optional
```
### Trait-Based Retrieval
For trait-based retrieval, please refer to the demo code below:
```bash
python code/trait_retrieval.py --support_image /path/to/sample/image.png \
  --support_mask /path/to/greyscale_mask.png \ 
  --trait_id 1 \ # target trait to retrieve, denote by the value in support mask  \
  --query_images /path/to/query/images/folder \
  --output /path/to/output/folder \
  --output_format "png" \ # png or gif, optional
  --top_k 5 # n top retrievals to save as results
```

## 📊 Dataset
Beetle part segmentation dataset is out! Available [here](https://github.com/Imageomics/NEON_beetles_masks.git).
Butterfly trait segmentation dataset can be accessed [here](https://github.com/ZiheHerzWang/butterfly_dataset/tree/main)!

## ❤️ Acknowledgements
This project makes use of the [SAM2](https://github.com/facebookresearch/sam2) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) codebases. We are grateful to the developers and maintainers of these projects for their contributions to the open-source community.
We thank [LoRA](https://github.com/microsoft/LoRA) for their great work.


## 📝 Citation
If you find our work helpful for your research, please consider citing using the following BibTeX entry:
```bibtex
@misc{feng2025staticsegmentationtrackingfrustratingly,
      title={Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation}, 
      author={Zhenyang Feng and Zihe Wang and Saul Ibaven Bueno and Tomasz Frelek and Advikaa Ramesh and Jingyan Bai and Lemeng Wang and Zanming Huang and Jianyang Gu and Jinsu Yoo and Tai-Yu Pan and Arpita Chowdhury and Michelle Ramirez and Elizabeth G. Campolongo and Matthew J. Thompson and Christopher G. Lawrence and Sydne Record and Neil Rosser and Anuj Karpatne and Daniel Rubenstein and Hilmar Lapp and Charles V. Stewart and Tanya Berger-Wolf and Yu Su and Wei-Lun Chao},
      year={2025},
      eprint={2501.06749},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.06749}, 
}
```
