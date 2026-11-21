
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load training manifest
df = pd.read_csv(PROJECT_ROOT / "data/processed/train_manifest.csv")

# Select first sample
sample = df.iloc[0]

# Load image, object mask and edge mask
image = Image.open(PROJECT_ROOT / sample["image_path"]).convert("RGB")
mask = Image.open(PROJECT_ROOT / sample["mask_path"]).convert("L")
edge = Image.open(PROJECT_ROOT / sample["edge_path"]).convert("L")

# Display all three
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Object Mask")

axes[2].imshow(edge, cmap="gray")
axes[2].set_title("Edge Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Image size:", image.size)
print("Mask size:", mask.size)
print("Edge size:", edge.size)
print("Mask pixel values:", np.unique(np.array(mask))[:20])
print("Edge pixel values:", np.unique(np.array(edge))[:20])
