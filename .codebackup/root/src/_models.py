# From C:\github\Tree-Canopy-Detection\src\models\sample_model_provided.py
import torch
import torch.nn as nn


class SampleModelProvided(nn.Module):
    """
    U-Net style segmentation model rewritten to match the format of models used.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_features=64,
        use_batchnorm=True,
        dropout=0.0,
    ):
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


# From C:\github\Tree-Canopy-Detection\src\models\simple_cnn.py
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
        out_channels: int = 1,
        features: int = 32,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
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
        x = self.encoder(x)
        logits = self.head(x)
        # return torch.sigmoid(logits)
        return logits


# From C:\github\Tree-Canopy-Detection\src\models\unet.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two layer convolution block used in U Net.
    Each conv is followed by batch norm and ReLU.
    """

    def __init__( self, in_ch: int, out_ch: int ):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace = True),

                nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace = True),
        )

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U Net for binary segmentation.
    Input is RGB. Output is one channel probability mask.
    """

    def __init__( self, in_channels: int = 3, out_channels: int = 1 ):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        b = self.bottleneck(self.pool(c4))

        u4 = self.up4(b)
        u4 = torch.cat([u4, c4], dim = 1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, c3], dim = 1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim = 1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim = 1)
        d1 = self.dec1(u1)

        out = self.out(d1)
        return torch.sigmoid(out)


# From C:\github\Tree-Canopy-Detection\src\models\zoo.py
from src.models.simple_cnn import SimpleCNN
from src.models.unet import UNet


try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

try:
    import timm

    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

try:
    from transformers import SegformerForSemanticSegmentation

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def create_simple_cnn(in_channels: int = 3, out_channels: int = 1):
    return SimpleCNN(in_channels=in_channels, out_channels=out_channels)


def create_unet(in_channels: int = 3, out_channels: int = 1):
    return UNet(in_channels=in_channels, out_channels=out_channels)


def create_smp_unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    out_channels=1,
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_fpn(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_linknet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.Linknet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_deeplabv3(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_deeplabv3plus(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_segformer(
    model_name="nvidia/segformer-b0-finetuned-ade-512-512", out_channels=1
):
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=out_channels,
    )
    return model


def create_timm_segformer(encoder_name="tf_efficientnetv2_s", out_channels=1):
    if not TIMM_AVAILABLE:
        raise ImportError("timm not installed")
    backbone = timm.create_model(
        encoder_name,
        features_only=True,
        pretrained=True,
        in_chans=3,
    )
    raise NotImplementedError("timm segformer head integration needs custom head")


def create_timm_upernet(encoder_name="swin_base_patch4_window7_224", out_channels=1):
    if not TIMM_AVAILABLE:
        raise ImportError("timm not installed")
    raise NotImplementedError(
        "UPerNet using timm backbone available if needed. Ask to enable."
    )


# Registry for building models (used by build_model)
MODEL_BUILDERS = {
    "simple_cnn": create_simple_cnn,
    "unet": create_unet,
    "smp_unet": create_smp_unet,
    "smp_fpn": create_smp_fpn,
    "smp_linknet": create_smp_linknet,
    "smp_deeplabv3": create_smp_deeplabv3,
    "smp_deeplabv3plus": create_smp_deeplabv3plus,
    "segformer": create_segformer,
}

# Registry for benchmarking (model kwargs)
MODEL_BENCHMARKS = {
    "simple_cnn": {},
    "unet": {},
    "smp_unet": {"encoder_name": "resnet34"},
    "smp_fpn": {"encoder_name": "resnet34"},
    "smp_linknet": {"encoder_name": "resnet34"},
    "smp_deeplabv3": {"encoder_name": "resnet34"},
    "smp_deeplabv3plus": {"encoder_name": "resnet34"},
    "segformer": {"model_name": "nvidia/segformer-b0-finetuned-ade-512-512"},
}


def build_model(name: str, **kwargs):
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name}")
    return MODEL_BUILDERS[name](**kwargs)


# From C:\github\Tree-Canopy-Detection\src\models\__init__.py


