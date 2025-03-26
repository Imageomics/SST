#%%
import json
import os
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

metadata_path = "/fs/scratch/PAS2099/danielf/SAM2/segment-anything-2/code/beetles/metadata.json"
masks_root = "/fs/scratch/PAS2099/danielf/SAM2/segment-anything-2/code/beetles/masks"

with open(metadata_path) as f:
    metadata = json.load(f)

urls = metadata["url"]
mask_paths = metadata["mask_path"]
species = metadata["species"]
cmap = plt.get_cmap("tab10")

demo_imgs = []

for url, mask_path, species in zip(urls, mask_paths, species):

    # download the image
    response = requests.get(url)
    assert response.status_code == 200, f"Failed to download {url}!"
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = np.array(img)

    # load the mask
    masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masks = cv2.resize(masks, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    masks = [masks == i for i in range(1, masks.max() + 1)]

    # overlay the masks
    for i, mask in enumerate(masks):
        img[mask] = np.array(cmap(i)[:3]) * 255

    demo_imgs.append(img)

# save the demo images as a gif
demo_imgs = [Image.fromarray(img).resize((256, 256)) for img in demo_imgs]
demo_imgs[0].save("demo.gif", save_all=True, append_images=demo_imgs[1:], duration=500, loop=0)
print(f"Demo gif saved to {os.path.abspath('demo.gif')}")