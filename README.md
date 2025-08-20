# Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation
[Imageomics Institute](https://imageomics.osu.edu/)

[Zhenyang Feng](https://defisch.github.io/), [Zihe Wang](https://ziheherzwang.github.io/HerzWangWebsite/), Saul Ibaven Bueno, Tomasz Frelek, Advikaa Ramesh, Jingyan Bai, Lemeng Wang, [Zanming Huang](https://tzmhuang.github.io/), [Jianyang Gu](https://vimar-gu.github.io/), [Jinsu Yoo](https://jinsuyoo.info/), [Tai-Yu Pan](https://tydpan.github.io/), Arpita Chowdhury, Michelle Ramirez, [Elizabeth G Campolongo](https://egrace479.github.io/), [Matthew J Thompson](https://www.linkedin.com/in/thompson-m-j/), [Christopher G. Lawrence](https://eeb.princeton.edu/people/christopher-lawrence), [Sydne Record](https://umaine.edu/wle/faculty-staff-directory/sydne-record/), [Neil Rosser](https://people.miami.edu/profile/74f02be76bd3ae57ed9edfdad0a3f76d), [Anuj Karpatne](https://anujkarpatne.github.io/), [Daniel Rubenstein](https://eeb.princeton.edu/people/daniel-rubenstein), [Hilmar Lapp](https://lappland.io/), [Charles V. Stewart](https://www.cs.rpi.edu/~stewart/), [Tanya Berger-Wolf](https://cse.osu.edu/people/berger-wolf.1), [Yu Su](https://ysu1989.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao)

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
Set `CUDA_HOME` to your cuda path (this is for grounding DINO)

For example:
```
export CUDA_HOME=/usr/local/cuda
```

Then sync uv packages:

```
uv sync
```

Download weights into checkpoints folder (you will need `wget`):

```
cd checkpoints
./download_ckpts.sh
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## 🧑‍💻 Usage

### Specimen Segmentation
Go to the [SAM](https://segment-anything.com/) demo, upload a representative image (e.g., `img001.png`), click the portions to segment, and select "Cut out object" from the sidebar. Right click and save the extraction (`img001_extracted.png`).

Then run the following two commands to generate the mask (like a guide for the model in segmentation shape--note the final processed image will _appear_ to be an all black image):

```
uv run python src/sst/get_mask_from_crop.py \
--image_path img001.png \
--image_crop_path img001_extracted.png \
--mask_image_path_out img001_extracted_processed.png
```


```
uv run python src/sst/prepare_starter_mask.py \
--mask_image_path img001_extracted_processed.png \
--mask_image_path_out img001_extracted_processed.png
```

Now that the mask has been generated, the following command can be run to segment your remaining images.

```
uv run python src/sst/segment_and_crop.py \
  --support_image img001.png \
  --support_mask img001_extracted_processed.png \
  --query_images [PATH_TO_IMAGE_DIRECTORY] \
  --output [PATH_TO_SEGMENTED_OUTPUT_DIRECTORY]
```


### Trait Segmentation
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
Beetle part segmentation dataset is available [here](data/neon_beetles/).

Butterfly trait segmentation dataset can be accessed [here](data/cambridge_butterfly/).

The instructions and appropriate citations for these datasets are provided in the Citation section of their respective READMEs.

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
