
import torch
import torch.nn as nn

from boundary_aware_unet import BoundaryAwareUNet
from dataloader import train_loader

device = torch.device("cpu")

# Create model
model = BoundaryAwareUNet().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Loss functions
bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(predictions, targets):
    predictions = torch.sigmoid(predictions)

    smooth = 1e-6

    intersection = (predictions * targets).sum()
    total = predictions.sum() + targets.sum()

    dice = (2 * intersection + smooth) / (total + smooth)

    return 1 - dice


# Get one training batch
batch = next(iter(train_loader))

images = batch["image"].to(device)
masks = batch["mask"].to(device)
edges = batch["edge"].to(device)

# Forward pass
segmentation, boundary = model(images)

# Calculate losses
loss_bce = bce_loss(segmentation, masks)
loss_dice = dice_loss(segmentation, masks)
loss_boundary = bce_loss(boundary, edges)

# Combined loss
total_loss = (
    loss_bce
    + loss_dice
    + 0.5 * loss_boundary
)

# Backpropagation
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("Boundary-aware training step completed!")
print("Segmentation BCE:", loss_bce.item())
print("Segmentation Dice:", loss_dice.item())
print("Boundary BCE:", loss_boundary.item())
print("Total Loss:", total_loss.item())
