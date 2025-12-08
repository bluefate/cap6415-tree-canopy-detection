import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two layer convolution block used in U Net.
    Each conv is followed by batch norm and ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Initialize a double convolution block.
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through double convolution block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after two conv-batchnorm-relu sequences.
        """
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U Net for binary segmentation.
    Input is RGB. Output is one channel probability mask.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """
        Initialize U-Net architecture.
        
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
            out_channels (int): Number of output channels for segmentation. Defaults to 3.
        """
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Encodes input through encoder path, processes through bottleneck,
        then decodes with skip connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output segmentation logits of shape [batch_size, out_channels, height, width].
        """
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        b = self.bottleneck(self.pool(c4))

        u4 = self.up4(b)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)

        out = self.out(d1)
        # return torch.sigmoid(out)
        return out
