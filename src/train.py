
from pathlib import Path
import torch
import torch.nn as nn

from unet import UNet
from dataloader import train_loader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use CPU for this initial test
device = torch.device("cpu")

# Create model
model = UNet().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# BCE Loss
bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(predictions, targets):
    predictions = torch.sigmoid(predictions)

    smooth = 1e-6

    intersection = (predictions * targets).sum()
    total = predictions.sum() + targets.sum()

    dice = (2 * intersection + smooth) / (total + smooth)

    return 1 - dice


# Get one batch only
batch = next(iter(train_loader))

images = batch["image"].to(device)
masks = batch["mask"].to(device)

# Forward pass
outputs = model(images)

# Calculate combined loss
loss_bce = bce_loss(outputs, masks)
loss_dice = dice_loss(outputs, masks)

loss = loss_bce + loss_dice

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step completed successfully!")
print("Batch shape:", images.shape)
print("BCE Loss:", loss_bce.item())
print("Dice Loss:", loss_dice.item())
print("Total Loss:", loss.item())
