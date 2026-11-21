
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = 256


class COD10KDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]
        edge_path = PROJECT_ROOT / row["edge_path"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        edge = Image.open(edge_path).convert("L")

        image = TF.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.NEAREST
        )

        edge = TF.resize(
            edge, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)
        mask = (TF.to_tensor(mask) > 0).float()
        edge = (TF.to_tensor(edge) > 0).float()

        return {
            "image": image,
            "mask": mask,
            "edge": edge
        }


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data/processed/train_manifest.csv"

    dataset = COD10KDataset(csv_path)
    sample = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Edge shape:", sample["edge"].shape)
