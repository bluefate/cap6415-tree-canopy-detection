import torch
import torch.nn as nn


class SampleModelProvided(nn.Module):
    """
    U-Net style segmentation model for semantic segmentation.
    
    Reference implementation with configurable batch normalization, dropout, and base features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features=64,
        use_batchnorm=True,
        dropout=0.0,
    ):
        """
        Initialize SampleModelProvided.
        
        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            out_channels (int): Number of output channels. Defaults to 3.
            base_features (int): Base number of features in convolution blocks. Defaults to 64.
            use_batchnorm (bool): Whether to use batch normalization. Defaults to True.
            dropout (float): Dropout rate. Defaults to 0.0.
        """
        super().__init__()

        def block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        f1 = base_features
        f2 = base_features * 2
        f3 = base_features * 4
        f4 = base_features * 8
        f5 = base_features * 16

        self.enc1 = block(in_channels, f1)
        self.enc2 = block(f1, f2)
        self.enc3 = block(f2, f3)
        self.enc4 = block(f3, f4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = block(f4, f5)

        self.up4 = nn.ConvTranspose2d(f5, f4, 2, 2)
        self.dec4 = block(f4 + f4, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, 2, 2)
        self.dec3 = block(f3 + f3, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, 2, 2)
        self.dec2 = block(f2 + f2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, 2, 2)
        self.dec1 = block(f1 + f1, f1)

        self.head = nn.Conv2d(f1, out_channels, 1)

    def forward(self, x):
        """
        Forward pass through U-Net architecture.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output segmentation logits with shape (B, num_classes, H, W).
        """
        c1 = self.enc1(x)
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

        logits = self.head(d1)
        # return torch.sigmoid(logits)
        return logits
