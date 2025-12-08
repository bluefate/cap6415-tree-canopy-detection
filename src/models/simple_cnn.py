import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Small convolutional model for binary segmentation.
    Good baseline before using UNet or larger encoder models.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 32,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize SimpleCNN model.
        
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
            out_channels (int): Number of output channels for segmentation. Defaults to 3.
            features (int): Base number of features in convolution filters. Defaults to 32.
            use_batchnorm (bool): Whether to use batch normalization. Defaults to True.
            dropout (float): Dropout rate (0-1). Defaults to 0.0.
        """
        super(SimpleCNN, self).__init__()

        def block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # lightweight feature extractor
        self.encoder = nn.Sequential(
            block(in_channels, features),
            block(features, features * 2),
            block(features * 2, features),
        )
        # segmentation head
        self.head = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output logits of shape [batch_size, out_channels, height, width].
        """
        x = self.encoder(x)
        logits = self.head(x)
        # return torch.sigmoid(logits)
        return logits
