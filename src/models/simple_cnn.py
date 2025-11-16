import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Small convolutional model for binary segmentation.
    Used in your original exploration and group or individual tree detection.
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            features: int = 32,
            use_batchnorm: bool = True,
            dropout: float = 0.0,
    ):
        super(SimpleCNN, self).__init__()

        def block( in_ch, out_ch ):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
                block(in_channels, features),
                block(features, features * 2),
                block(features * 2, features),
        )


    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        x = self.encoder(x)
        logits = self.head(x)
        return torch.sigmoid(logits)
