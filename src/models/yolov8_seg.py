import torch
import torch.nn as nn


# Check if ultralytics is available
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


# =============================================================================
# APPROACH 1: PyTorch Wrapper (integrates with existing training loop)
# =============================================================================


class YOLOv8SegmentationWrapper(nn.Module):
    """
    Wrapper around YOLOv8-seg that outputs semantic segmentation masks
    compatible with CrossEntropyLoss and the existing training pipeline.

    This wrapper:
    - Takes standard image tensors [B, 3, H, W]
    - Outputs segmentation logits [B, num_classes, H, W]
    - Can be trained with standard PyTorch optimizers

    Args:
        model_size: One of 'n', 's', 'm', 'l', 'x' for nano to extra-large
        num_classes: Number of output classes (default 3: bg, individual, group)
        pretrained: Whether to load pretrained weights
    """

    def __init__(
        self,
        model_size: str = "n",
        num_classes: int = 3,
        pretrained: bool = True,
        image_size: int = 640,
    ):
        super().__init__()

        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required for YOLOv8. "
                "Install with: pip install ultralytics"
            )

        self.num_classes = num_classes
        self.image_size = image_size
        self.model_size = model_size

        # Load YOLOv8-seg model
        model_name = f"yolov8{model_size}-seg.pt"
        self.yolo = YOLO(model_name)

        # Get the underlying PyTorch model
        self.backbone = self.yolo.model

        # Modify the segmentation head for our number of classes
        # YOLOv8-seg has a complex head structure, we'll add an adapter
        self._adapt_for_semantic_segmentation()

    def _adapt_for_semantic_segmentation(self):
        """
        Adapt YOLOv8's instance segmentation to semantic segmentation.

        YOLOv8-seg outputs instance masks, but we need semantic masks.
        We add a semantic segmentation head on top.
        """
        # YOLOv8 backbone feature dimensions vary by size
        # We'll use the P3 feature map (stride 8) for segmentation
        feature_dims = {
            "n": 64,  # nano
            "s": 128,  # small
            "m": 192,  # medium
            "l": 256,  # large
            "x": 320,  # extra-large
        }

        in_channels = feature_dims.get(self.model_size, 128)

        # Semantic segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1),
        )

        # Freeze backbone initially (optional, can be unfrozen for fine-tuning)
        self.freeze_backbone = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for semantic segmentation.

        Args:
            x: Input tensor [B, 3, H, W], values in [0, 1]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape

        # YOLOv8 expects images in a specific format
        # Scale to 0-255 range if needed
        if x.max() <= 1.0:
            x = x * 255.0

        # Get backbone features
        if self.freeze_backbone:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)

        # Apply segmentation head
        seg_logits = self.seg_head(features)

        # Upsample to original resolution
        seg_logits = torch.nn.functional.interpolate(
            seg_logits, size=(H, W), mode="bilinear", align_corners=False
        )

        return seg_logits

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from YOLOv8 backbone.

        We need to hook into the backbone to get intermediate features.
        """
        # Use the backbone's forward method
        # YOLOv8 model structure: backbone -> neck -> head

        # Simple approach: run through model and capture features
        # This is a simplified version - full implementation would hook specific layers

        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output

        # Register hook on an early layer
        # The exact layer depends on model architecture
        try:
            # Try to hook into backbone
            handle = self.backbone.model[4].register_forward_hook(hook_fn)
            _ = self.backbone(x)
            handle.remove()

            if features is not None:
                return features
        except:
            pass

        # Fallback: create simple feature extractor
        return self._simple_feature_extract(x)

    def _simple_feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple feature extraction fallback.
        """
        # Simplified backbone for compatibility
        if not hasattr(self, "_fallback_backbone"):
            in_ch = 3
            out_ch = 128
            self._fallback_backbone = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ).to(x.device)

        return self._fallback_backbone(x / 255.0 if x.max() > 1 else x)


# =============================================================================
# APPROACH 2: Simplified YOLOv8 Semantic Segmentation (Recommended)
# =============================================================================


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
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding=k // 2 if p is None else p, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
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
