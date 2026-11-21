
from pathlib import Path
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from dataset import COD10KDataset
from unet import UNet
from attention_unet import AttentionUNet
from boundary_aware_unet import BoundaryAwareUNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Choose model: "unet", "attention", or "boundary"
MODEL_NAME = "unet"

BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.10
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)
random.seed(SEED)


def dice_loss(predictions, targets):
    predictions = torch.sigmoid(predictions)

    smooth = 1e-6
    intersection = (predictions * targets).sum()
    total = predictions.sum() + targets.sum()

    dice = (2 * intersection + smooth) / (total + smooth)
    return 1 - dice


def get_model(name):
    if name == "unet":
        return UNet()

    if name == "attention":
        return AttentionUNet()

    if name == "boundary":
        return BoundaryAwareUNet()

    raise ValueError("Unknown model name")


def calculate_loss(model, batch, bce):
    images = batch["image"].to(DEVICE)
    masks = batch["mask"].to(DEVICE)
    edges = batch["edge"].to(DEVICE)

    if MODEL_NAME == "boundary":
        segmentation, boundary = model(images)

        seg_bce = bce(segmentation, masks)
        seg_dice = dice_loss(segmentation, masks)
        boundary_bce = bce(boundary, edges)

        loss = seg_bce + seg_dice + 0.5 * boundary_bce

    else:
        segmentation = model(images)

        seg_bce = bce(segmentation, masks)
        seg_dice = dice_loss(segmentation, masks)

        loss = seg_bce + seg_dice

    return loss


def main():
    print("Device:", DEVICE)
    print("Selected model:", MODEL_NAME)

    dataset = COD10KDataset(
        PROJECT_ROOT / "data/processed/train_manifest.csv"
    )

    val_size = int(len(dataset) * VALIDATION_RATIO)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = get_model(MODEL_NAME).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    bce = nn.BCEWithLogitsLoss()

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)

    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            loss = calculate_loss(model, batch, bce)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}",
                    flush=True
                )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                loss = calculate_loss(model, batch, bce)
                val_loss += loss.item()

        val_loss /= len(val_loader)


        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_name": MODEL_NAME,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss
                },
                checkpoint_dir / f"{MODEL_NAME}_best.pth"
            )

            print("Best model checkpoint saved!")

    print("Training completed!")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
