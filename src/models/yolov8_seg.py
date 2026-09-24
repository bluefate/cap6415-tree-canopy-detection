import torch
import torch.nn as nn


# Check if ultralytics is available
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class YOLOv8SemanticSeg(nn.Module):
    """
    Simplified YOLOv8-style architecture for semantic segmentation.

    This is a clean implementation that:
    - Uses YOLOv8's CSPDarknet-style backbone
    - Adds FPN-style neck
    - Outputs semantic segmentation masks

    Much more reliable than wrapping the detection model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.25,
    ):
        """
        Initialize YOLOv8-style semantic segmentation model.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            out_channels (int): Number of output classes. Defaults to 3.
            base_channels (int): Base channel count. Defaults to 32.
            depth_multiple (float): Depth scaling factor. Defaults to 0.33.
            width_multiple (float): Width scaling factor. Defaults to 0.25.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate channel sizes based on width multiple
        c1 = int(64 * width_multiple)
        c2 = int(128 * width_multiple)
        c3 = int(256 * width_multiple)
        c4 = int(512 * width_multiple)
        c5 = int(1024 * width_multiple)

        # Stem
        self.stem = Conv(in_channels, c1, k=3, s=2)

        # Backbone (CSPDarknet-style)
        self.stage1 = nn.Sequential(
            Conv(c1, c2, k=3, s=2),
            C2f(c2, c2, n=max(1, int(3 * depth_multiple))),
        )

        self.stage2 = nn.Sequential(
            Conv(c2, c3, k=3, s=2),
            C2f(c3, c3, n=max(1, int(6 * depth_multiple))),
        )

        self.stage3 = nn.Sequential(
            Conv(c3, c4, k=3, s=2),
            C2f(c4, c4, n=max(1, int(6 * depth_multiple))),
        )

        self.stage4 = nn.Sequential(
            Conv(c4, c5, k=3, s=2),
            C2f(c5, c5, n=max(1, int(3 * depth_multiple))),
            SPPF(c5, c5),
        )

        # FPN-style decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral4 = Conv(c5, c4, k=1)
        self.fpn4 = C2f(c4 + c4, c4, n=max(1, int(3 * depth_multiple)))

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral3 = Conv(c4, c3, k=1)
        self.fpn3 = C2f(c3 + c3, c3, n=max(1, int(3 * depth_multiple)))

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral2 = Conv(c3, c2, k=1)
        self.fpn2 = C2f(c2 + c2, c2, n=max(1, int(3 * depth_multiple)))

        # Segmentation head
        self.seg_head = nn.Sequential(
            Conv(c2, c2, k=3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv(c2, c1, k=3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [B, 3, H, W]

        Returns:
            Segmentation logits [B, out_channels, H, W]
        """
        # Encoder
        x0 = self.stem(x)  # /2
        x1 = self.stage1(x0)  # /4
        x2 = self.stage2(x1)  # /8
        x3 = self.stage3(x2)  # /16
        x4 = self.stage4(x3)  # /32

        # FPN Decoder
        p4 = self.lateral4(x4)
        p4 = self.up4(p4)
        p4 = torch.cat([p4, x3], dim=1)
        p4 = self.fpn4(p4)

        p3 = self.lateral3(p4)
        p3 = self.up3(p3)
        p3 = torch.cat([p3, x2], dim=1)
        p3 = self.fpn3(p3)

        p2 = self.lateral2(p3)
        p2 = self.up2(p2)
        p2 = torch.cat([p2, x1], dim=1)
        p2 = self.fpn2(p2)

        # Segmentation head
        out = self.seg_head(p2)

        return out


# =============================================================================
# YOLOv8 Building Blocks
# =============================================================================


class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize convolution block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size. Defaults to 1.
            s (int): Stride. Defaults to 1.
            p (int, optional): Padding. If None, uses k//2.
            g (int): Groups. Defaults to 1.
            act (bool): Use activation. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding=k // 2 if p is None else p, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch norm, and activation."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Initialize bottleneck block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Use residual connection. Defaults to True.
            g (int): Groups for grouped convolution. Defaults to 1.
            e (float): Expansion ratio. Defaults to 0.5.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional residual connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of bottleneck blocks. Defaults to 1.
            shortcut (bool): Use shortcut connections. Defaults to False.
            g (int): Groups for grouped convolution. Defaults to 1.
            e (float): Expansion ratio. Defaults to 0.5.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Apply CSP bottleneck with cross-stage feature fusion."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5):
        """
        Initialize Spatial Pyramid Pooling - Fast module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size for max pooling. Defaults to 5.
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply spatial pyramid pooling with multiple scales."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# =============================================================================
# Factory Functions
# =============================================================================


def create_yolov8_seg(
    in_channels: int = 3,
    out_channels: int = 3,
    model_size: str = "n",
) -> nn.Module:
    """
    Create YOLOv8-style semantic segmentation model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        model_size: One of 'n', 's', 'm', 'l', 'x'

    Returns:
        PyTorch model
    """
    # Model scaling parameters (from YOLOv8)
    size_configs = {
        "n": {"depth": 0.33, "width": 0.25},  # nano
        "s": {"depth": 0.33, "width": 0.50},  # small
        "m": {"depth": 0.67, "width": 0.75},  # medium
        "l": {"depth": 1.00, "width": 1.00},  # large
        "x": {"depth": 1.00, "width": 1.25},  # extra-large
    }

    if model_size not in size_configs:
        raise ValueError(f"model_size must be one of {list(size_configs.keys())}")

    config = size_configs[model_size]

    return YOLOv8SemanticSeg(
        in_channels=in_channels,
        out_channels=out_channels,
        depth_multiple=config["depth"],
        width_multiple=config["width"],
    )


def create_yolov8n_seg(in_channels: int = 3, out_channels: int = 3) -> nn.Module:
    """YOLOv8-nano for segmentation."""
    return create_yolov8_seg(in_channels, out_channels, "n")


def create_yolov8s_seg(in_channels: int = 3, out_channels: int = 3) -> nn.Module:
    """YOLOv8-small for segmentation."""
    return create_yolov8_seg(in_channels, out_channels, "s")


def create_yolov8m_seg(in_channels: int = 3, out_channels: int = 3) -> nn.Module:
    """YOLOv8-medium for segmentation."""
    return create_yolov8_seg(in_channels, out_channels, "m")


def create_yolov8l_seg(in_channels: int = 3, out_channels: int = 3) -> nn.Module:
    """YOLOv8-large for segmentation."""
    return create_yolov8_seg(in_channels, out_channels, "l")


# =============================================================================
# Test Function
# =============================================================================


def test_yolov8_seg():
    """Test the YOLOv8 segmentation model."""
    print("Testing YOLOv8 Semantic Segmentation...")

    # Test different sizes
    for size in ["n", "s", "m"]:
        model = create_yolov8_seg(in_channels=3, out_channels=3, model_size=size)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            y = model(x)

        print(
            f"YOLOv8-{size}: params={params/1e6:.2f}M, "
            f"input={tuple(x.shape)}, output={tuple(y.shape)}"
        )

        # Verify output shape
        assert y.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {y.shape}"

    print("All tests passed!")


if __name__ == "__main__":
    test_yolov8_seg()
