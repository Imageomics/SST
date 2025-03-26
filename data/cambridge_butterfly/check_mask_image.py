#%%
#%%
import json
import random
import os
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the path to the JSON file
json_file_path = "train_test_separate/cyrbia/test_data.json"

# Load the JSON data
with open(json_file_path, "r") as file:
    data = json.load(file)

# Randomly select 10 pairs to demonstrate how to read and use the dataset
random_elements = random.sample(data, 10)

# Use a colormap from matplotlib
cmap = plt.get_cmap("tab20")  # Use 'tab20' for up to 20 distinct colors

# Process each selected element
for element in random_elements:
    x_value = element[0]
    image_url = element[1]  # URL to get the image
    mask_relative_path = element[2]  # Relative path to get the mask

    try:
        # Step 1: Download the image
        print(f"Loading image from URL: {image_url}")
        with urllib.request.urlopen(image_url) as response:
            img = Image.open(response).convert("RGB")
            img = np.array(img)  # Convert to NumPy array for easier manipulation

        # Step 2: Load the mask
        print(f"Loading mask from relative path: {mask_relative_path}")
        if not os.path.exists(mask_relative_path):
            raise FileNotFoundError(f"Mask file not found: {mask_relative_path}")
        mask = Image.open(mask_relative_path).convert("L")
        mask = np.array(mask)  # Convert to NumPy array

        # Step 3: Resize the mask to match the image size
        mask_resized = np.array(
            Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        )

        # Step 4: Apply a colormap to the mask
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        for i in range(1, mask_resized.max() + 1):  # Skip the background (0)
            colored_mask[mask_resized == i] = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)

        # Step 5: Overlay the mask on the image
        overlay = img.copy()
        alpha = 0.5  # Transparency level
        mask_indices = mask_resized > 0  # Non-zero mask regions
        overlay[mask_indices] = (
            alpha * colored_mask[mask_indices] + (1 - alpha) * img[mask_indices]
        ).astype(np.uint8)

        # Step 6: Display the overlayed image
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"Overlayed Image (X Value: {x_value})")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error processing {image_url} or {mask_relative_path}: {e}")
