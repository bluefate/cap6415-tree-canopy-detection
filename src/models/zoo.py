from src.models.simple_cnn import SimpleCNN
from src.models.unet import UNet


# Model	            Year	Key Idea	            Strengths	                        Weaknesses
# ResNet	        2015	Residual connections	Robust, widely used	                Heavy, less efficient
# EfficientNet-B4	2019	Compound scaling	    High accuracy per parameter	        Larger input size
# EfficientNetV2	2021	Faster scaling	        Efficient training	                Still CNN-based
# ViT	            2020	Pure transformer	    Scales well, high accuracy	        Needs huge datasets
# Swin Transformer	2021	Shifted windows	        Great for segmentation/detection	More complex
# ConvNeXt	        2022	Modern CNN	            Efficient, strong accuracy          Less novel than ViTs

# (ConvNeXt Variants)
# Variant	        Params	Use Case
# convnext_tiny	    ~28M	Lightweight, fast training, good for smaller datasets or limited GPU
# convnext_base	    ~89M	Balanced accuracy vs compute, strong general-purpose backbone
# convnext_large	~198M	High accuracy, but heavy â€” requires strong GPUs
# convnext_xlarge	~350M	State-of-the-art accuracy, but very resource-intensive


# out_channels=3, # <- Should be 3, background=0, individual_tree=1, group_of_trees=2

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


def create_simple_cnn(in_channels: int = 3, out_channels: int = 3):
    return SimpleCNN(in_channels=in_channels, out_channels=out_channels)


def create_unet(in_channels: int = 3, out_channels: int = 3):
    return UNet(in_channels=in_channels, out_channels=out_channels)


def create_smp_unet(
    encoder_name="convnext_tiny",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
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
    encoder_name="convnext_tiny",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
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
    encoder_name="convnext_tiny",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
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
    encoder_name="convnext_tiny",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
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
    encoder_name="convnext_tiny",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
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
    model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    in_channels: int = 3,
    out_channels: int = 3,
):
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed")
    # Note: SegFormer does NOT support custom in_channels.
    # It assumes 3-channel RGB input. We ignore in_channels to prevent errors.
    if in_channels != 3:
        raise ValueError("SegFormer only supports 3-channel RGB input.")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=out_channels, ignore_mismatched_sizes=True
    )

    return model


def create_timm_segformer(encoder_name="tf_efficientnetv2_s", out_channels: int = 3):
    if not TIMM_AVAILABLE:
        raise ImportError("timm not installed")
    backbone = timm.create_model(
        encoder_name,
        features_only=True,
        pretrained=True,
        in_chans=3,
    )
    raise NotImplementedError("timm segformer head integration needs custom head")


def create_timm_upernet(
    encoder_name="swin_base_patch4_window7_224", out_channels: int = 3
):
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
