
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import COD10KDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_SIZE = 4

# Load training dataset
train_dataset = COD10KDataset(
    PROJECT_ROOT / "data/processed/train_manifest.csv"
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

if __name__ == "__main__":
    images_seen = 0

    for batch in train_loader:
        images = batch["image"]
        masks = batch["mask"]
        edges = batch["edge"]

        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        print("Edge batch shape:", edges.shape)

        images_seen += images.shape[0]
        break

    print("Images in first batch:", images_seen)
    print("Total training samples:", len(train_dataset))
    print("DataLoader working successfully!")
