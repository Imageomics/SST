# Cambridge Butterfly Trait Masks 🦋

This repository provides a dataset for butterfly species classification and segmentation. The dataset is structured for standard and one-shot segmentation tasks. A Python script is included to demonstrate how to read and visualize masks on butterfly images.

## Dataset Structure

### `Major_species_data`
- Contains data for major butterfly species and hybrids, there are at least 250 annotated images in each species.
- Subdirectories represent specific species or hybrid groups, such as:
  - `(malletti x plesseni) x malleti`
  - `cyrbia`
  - `lativitta`
  - `malletti`
  - `notabilis x lativitta`

### `Minor_species_data`
- Contains data for less prominent butterfly species, which can be used for one-shot segmentation tasks. Each species contain at least 2 samples, suitable for one-shot segmentation task.

### `train_test_separate`
- Includes pre-split data for training and testing purposes.
- Subdirectories correspond to species or hybrid groups and contain:
  - `train_data.json`: Metadata and paths for training images.
  - `test_data.json`: Metadata and paths for testing images.

### Supporting Files
- **`minor_species.json`**: 
  - Contains metadata or supplementary details about minor butterfly species in the dataset.
  - Includes annotations for left-right symmetry patterns, allowing for simplified mask merging based on symmetry mappings (e.g., `1:0` means the 1st mask belongs to mask pair 0).

- **`check_mask_image.py`**: Python script for validating and visualizing image-mask overlays.

---

## Code Description

### `check_mask_image.py`

This Python script demonstrates how to read the dataset, download images, and overlay corresponding segmentation masks. It performs the following tasks:

#### 1. Load the Dataset
- Reads a JSON file (e.g., `train_test_separate/cyrbia/test_data.json`) containing image URLs and mask paths.
- Randomly selects 10 pairs for demonstration.

#### 2. Download Images
- Downloads images from the URLs provided in the JSON file.
- Converts images to RGBA format for easier overlay processing.

#### 3. Load and Resize Masks
- Loads grayscale masks from the relative paths specified in the JSON file.
- Resizes masks to match the size of their corresponding images.

#### 4. Apply Colors to Masks
- Uses a colormap (`matplotlib`'s `tab20`) to programmatically assign distinct colors to mask regions.

#### 5. Overlay Masks onto Images
- Combines the colored mask with the original image using alpha transparency for visualization.
- Displays the overlayed images using Matplotlib.

#### 6. Error Handling
- Ensures proper file existence and handles mismatches in image and mask sizes.

---
## Citation
If you find this dataset useful, we kindly ask that, in addition to citing [our original work](../../README.md#-citation), you also cite the following sources. We sincerely appreciate their contribution in providing the original specimen images, which made this study possible.
```bibtex
@misc{lawrence2024heliconius,
  author       = {Christopher Lawrence and Elizabeth G. Campolongo and Neil Rosser},
  title        = {Heliconius Collection (Cambridge Butterfly)},
  year         = {2024},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/imageomics/Heliconius-Collection_Cambridge-Butterfly}
}

@misc{gabriela_montejo_kovacevich_2020_4289223,
author = {Gabriela Montejo-Kovacevich and Eva van der Heijden and Nicola Nadeau and Chris Jiggins},
title = {Cambridge butterfly wing collection batch 10},
month = nov,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4289223},
url = {https://doi.org/10.5281/zenodo.4289223}
}

@misc{patricio_a_salazar_2020_4288311,
author = {Patricio A. Salazar and Nicola Nadeau and Gabriela Montejo-Kovacevich and Chris Jiggins},
title = {{Sheffield butterfly wing collection - Patricio Salazar, Nicola Nadeau, Ikiam broods batch 1 and 2}},
month = nov,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4288311},
url = {https://doi.org/10.5281/zenodo.4288311} 
}

@misc{montejo_kovacevich_2019_2677821,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian},
title = {Cambridge butterfly wing collection batch 2},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2677821},
url = {https://doi.org/10.5281/zenodo.2677821} 
}

@misc{jiggins_2019_2682458,
author = {Jiggins, Chris and Montejo-Kovacevich, Gabriela and Warren, Ian and Wiltshire, Eva},
title = {Cambridge butterfly wing collection batch 3},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2682458},
url = {https://doi.org/10.5281/zenodo.2682458} 
}

@misc{montejo_kovacevich_2019_2682669,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian},
title = {Cambridge butterfly wing collection batch 4},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2682669},
url = {https://doi.org/10.5281/zenodo.2682669} 
}

@misc{montejo_kovacevich_2019_2684906,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Wiltshire, Eva},
title = {Cambridge butterfly wing collection batch 5},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2684906},
url = {https://doi.org/10.5281/zenodo.2684906} 
}

@misc{warren_2019_2552371,
author = {Warren, Ian and Jiggins, Chris},
title = {{Miscellaneous Heliconius wing photographs (2001-2019) Part 1}},
month = feb,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2552371},
url = {https://doi.org/10.5281/zenodo.2552371} 
}

@misc{warren_2019_2553977,
author = {Warren, Ian and Jiggins, Chris},
title = {{Miscellaneous Heliconius wing photographs (2001-2019) Part 3}},
month = feb,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2553977},
url = {https://doi.org/10.5281/zenodo.2553977} 
}

@misc{montejo_kovacevich_2019_2686762,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Wiltshire, Eva},
title = {Cambridge butterfly wing collection batch 6},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2686762},
url = {https://doi.org/10.5281/zenodo.2686762} 
}

@misc{jiggins_2019_2549524,
author = {Jiggins, Chris and Warren, Ian},
title = {{Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 1}},
month = jan,
year = 2019,
publisher = {Zenodo},
version = 1,
doi = {10.5281/zenodo.2549524},
url = {https://doi.org/10.5281/zenodo.2549524}
}

@misc{jiggins_2019_2550097,
author = {Jiggins, Chris and Warren, Ian},
title = {{Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 2}},
month = jan,
year = 2019,
publisher = {Zenodo},
version = 1,
doi = {10.5281/zenodo.2550097},
url = {https://doi.org/10.5281/zenodo.2550097}
}

@misc{joana_i_meier_2020_4153502,
author = {Joana I. Meier and Patricio Salazar and Gabriela Montejo-Kovacevich and Ian Warren and Chris Jggins},
title = {{Cambridge butterfly wing collection - Patricio Salazar PhD wild specimens batch 3}},
month = oct,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4153502},
url = {https://doi.org/10.5281/zenodo.4153502}
}

@misc{montejo_kovacevich_2019_3082688,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian},
title = {{Cambridge butterfly wing collection batch 1- version 2}},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.3082688},
url = {https://doi.org/10.5281/zenodo.3082688} 
}

@misc{montejo_kovacevich_2019_2813153,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Salazar, Camilo and Elias, Marianne and Gavins, Imogen and Wiltshire, Eva and Montgomery, Stephen and McMillan, Owen},
title = {{Cambridge and collaborators butterfly wing collection batch 10}},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2813153},
url = {https://doi.org/10.5281/zenodo.2813153} 
}

@misc{salazar_2018_1748277,
author = {Salazar, Patricio and Montejo-Kovacevich, Gabriela and Warren, Ian and Jiggins, Chris},
title = {{Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 1}},
month = dec,
year = 2018,
publisher = {Zenodo},
doi = {10.5281/zenodo.1748277},
url = {https://doi.org/10.5281/zenodo.1748277} 
}

@misc{montejo_kovacevich_2019_2702457,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Wiltshire, Eva},
title = {Cambridge butterfly wing collection batch 7},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2702457},
url = {https://doi.org/10.5281/zenodo.2702457}
}

@misc{salazar_2019_2548678,
author = {Salazar, Patricio and Montejo-Kovacevich, Gabriela and Warren, Ian and Jiggins, Chris},
title = {{Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 2}},
month = jan,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2548678},
url = {https://doi.org/10.5281/zenodo.2548678}
}

@misc{pinheiro_de_castro_2022_5561246,
author = {Pinheiro de Castro, Erika and Jiggins, Christopher and Lucas da Silva-Brand\u00e3o, Karina and Victor Lucci Freitas, Andre and Zikan Cardoso, Marcio and Van Der Heijden, Eva and Meier, Joana and Warren, Ian},
title = {{Brazilian Butterflies Collected December 2020 to January 2021}},
month = feb,
year = 2022,
publisher = {Zenodo},
doi = {10.5281/zenodo.5561246},
url = {https://doi.org/10.5281/zenodo.5561246}
}

@misc{montejo_kovacevich_2019_2707828,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Wiltshire, Eva},
title = {Cambridge butterfly wing collection batch 8},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2707828},
url = {https://doi.org/10.5281/zenodo.2707828}
}

@misc{montejo_kovacevich_2019_2714333,
author = {Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Wiltshire, Eva and Gavins, Imogen},
title = {Cambridge butterfly wing collection batch 9},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2714333},
url = {https://doi.org/10.5281/zenodo.2714333} 
}

@misc{gabriela_montejo_kovacevich_2020_4291095,
author = {Gabriela Montejo-Kovacevich and Eva van der Heijden and Chris Jiggins},
title = {{Cambridge butterfly collection - GMK Broods Ikiam 2018}},
month = nov,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4291095},
url = {https://doi.org/10.5281/zenodo.4291095}
}

@misc{gabriela_montejo_kovacevich_2019_3569598,
author = {Gabriela Montejo-Kovacevich and Letitia Cookson and Eva van der Heijden and Ian Warren and David P. Edwards and Chris Jiggins},
title = {Cambridge butterfly collection - Loreto, Peru 2018},
month = dec,
year = 2019,
publisher = {Zenodo},
version = {1.0.0},
doi = {10.5281/zenodo.3569598},
url = {https://doi.org/10.5281/zenodo.3569598}
}

@misc{gabriela_montejo_kovacevich_2020_4287444,
author = {Gabriela Montejo-Kovacevich and Letitia Cookson and Eva van der Heijden and Ian Warren and David P. Edwards and Chris Jiggins},
title = {{Cambridge butterfly collection - Loreto, Peru 2018 batch2}},
month = nov,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4287444},
url = {https://doi.org/10.5281/zenodo.4287444}
}

@misc{gabriela_montejo_kovacevich_2020_4288250,
author = {Gabriela Montejo-Kovacevich and Letitia Cookson and Eva van der Heijden and Ian Warren and David P. Edwards and Chris Jiggins},
title = {{Cambridge butterfly collection - Loreto, Peru 2018 batch3}},
month = nov,
year = 2020,
publisher = {Zenodo},
doi = {10.5281/zenodo.4288250},
url = {https://doi.org/10.5281/zenodo.4288250} 
}

@misc{montejo_kovacevich_2021_5526257,
author = {Montejo-Kovacevich, Gabriela and Paynter, Quentin and Ghane, Amin},
title = {{Heliconius erato cyrbia, Cook Islands (New Zealand) 2016, 2019, 2021}},
month = sep,
year = 2021,
publisher = {Zenodo},
doi = {10.5281/zenodo.5526257},
url = {https://doi.org/10.5281/zenodo.5526257}
}

@misc{warren_2019_2553501,
author = {Warren, Ian and Jiggins, Chris},
title = {{Miscellaneous Heliconius wing photographs (2001-2019) Part 2}},
month = feb,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2553501},
url = {https://doi.org/10.5281/zenodo.2553501} 
}

@misc{salazar_2019_2735056,
author = {Salazar, Camilo and Montejo-Kovacevich, Gabriela and Jiggins, Chris and Warren, Ian and Gavins, Imogen},
title = {{Camilo Salazar and Cambridge butterfly wing collection batch 1}},
month = may,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2735056},
url = {https://doi.org/10.5281/zenodo.2735056} 
}

@misc{mattila_2019_2554218,
author = {Mattila, Anniina and Jiggins, Chris and Warren, Ian},
title = {{University of Helsinki butterfly wing collection - Anniina Mattila field caught specimens}},
month = feb,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2554218},
url = {https://doi.org/10.5281/zenodo.2554218} 
}

@misc{mattila_2019_2555086,
author = {Mattila, Anniina and Jiggins, Chris and Warren, Ian},
title = {{University of Helsinki butterfly collection - Anniina Mattila bred specimens}},
month = feb,
year = 2019,
publisher = {Zenodo},
doi = {10.5281/zenodo.2555086},
url = {https://doi.org/10.5281/zenodo.2555086} 
}

```