# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pillow",
#     "term-image",
#     "tyro",
# ]
# ///
import pathlib

import numpy as np
import term_image.image
import tyro
from PIL import Image

colors = np.array([
    [0, 18, 25],
    [0, 95, 115],
    [10, 147, 150],
    [148, 210, 189],
    [233, 216, 166],
    [238, 155, 0],
    [202, 103, 2],
    [187, 62, 3],
    [174, 32, 18],
    [155, 34, 38],
])


def main(img: pathlib.Path, /):
    img = Image.open(img)
    w, h = img.size

    mono = np.array(img)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for color, i in zip(colors, sorted(np.unique(mono).tolist())):
        rgb[mono == i, :] = color
    img = term_image.image.AutoImage(Image.fromarray(rgb))
    img.draw()


if __name__ == "__main__":
    tyro.cli(main)
