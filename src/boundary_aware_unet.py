
import torch
import torch.nn as nn

from attention_unet import AttentionGate
from unet import DoubleConv


class BoundaryAwareUNet(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder with attention
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = DoubleConv(64, 32)

        # Two output heads
        self.segmentation_head = nn.Conv2d(32, 1, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(
            torch.cat([d4, self.att4(d4, e4)], dim=1)
        )

        d3 = self.up3(d4)
        d3 = self.dec3(
            torch.cat([d3, self.att3(d3, e3)], dim=1)
        )

        d2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat([d2, self.att2(d2, e2)], dim=1)
        )

        d1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat([d1, self.att1(d1, e1)], dim=1)
        )

        # Separate predictions
        segmentation = self.segmentation_head(d1)
        boundary = self.boundary_head(d1)

        return segmentation, boundary


if __name__ == "__main__":

    model = BoundaryAwareUNet()

    dummy_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        segmentation, boundary = model(dummy_input)

    print("Boundary-Aware Attention U-Net created successfully!")
    print("Input shape:", dummy_input.shape)
    print("Segmentation output:", segmentation.shape)
    print("Boundary output:", boundary.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
