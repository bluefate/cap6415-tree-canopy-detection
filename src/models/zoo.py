from src.models.simple_cnn import SimpleCNN
from src.models.unet import UNet
from src.utils.helpers import p, t


try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False


try:
    from transformers import SegformerForSemanticSegmentation

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from src.models.yolov8_seg import (
        create_yolov8_seg,
        create_yolov8n_seg,
        create_yolov8s_seg,
        create_yolov8m_seg,
        create_yolov8l_seg,
        YOLOv8SemanticSeg,
    )

    YOLO_SEG_AVAILABLE = True
except ImportError:
    YOLO_SEG_AVAILABLE = False


def create_simple_cnn(in_channels: int = 3, out_channels: int = 3):
    """
    Create a SimpleCNN model for baseline segmentation.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        SimpleCNN: Initialized model.
    """
    return SimpleCNN(in_channels=in_channels, out_channels=out_channels)


def create_unet(in_channels: int = 3, out_channels: int = 3):
    """
    Create a U-Net model for semantic segmentation.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        UNet: Initialized model.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels)


def create_smp_unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Create a segmentation_models_pytorch U-Net with configurable encoder.

    Args:
        encoder_name (str): Name of the encoder backbone. Defaults to "mobilenet_v2".
        encoder_weights (str): Pre-trained weights to load. Defaults to "imagenet".
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        smp.Unet: Initialized model.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_fpn(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Create a Feature Pyramid Network (FPN) model.

    Args:
        encoder_name (str): Name of the encoder backbone. Defaults to "mobilenet_v2".
        encoder_weights (str): Pre-trained weights to load. Defaults to "imagenet".
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        smp.FPN: Initialized model.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_linknet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Create a LinkNet model for fast semantic segmentation.

    Args:
        encoder_name (str): Name of the encoder backbone. Defaults to "mobilenet_v2".
        encoder_weights (str): Pre-trained weights to load. Defaults to "imagenet".
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        smp.Linknet: Initialized model.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.Linknet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_deeplabv3(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Create a DeepLabV3 model for semantic segmentation.

    Args:
        encoder_name (str): Name of the encoder backbone. Defaults to "mobilenet_v2".
        encoder_weights (str): Pre-trained weights to load. Defaults to "imagenet".
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        smp.DeepLabV3: Initialized model.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_smp_deeplabv3plus(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Create a DeepLabV3+ model for semantic segmentation.

    Args:
        encoder_name (str): Name of the encoder backbone. Defaults to "mobilenet_v2".
        encoder_weights (str): Pre-trained weights to load. Defaults to "imagenet".
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        smp.DeepLabV3Plus: Initialized model.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
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
    """
    Create a SegFormer model from Hugging Face transformers.

    Args:
        model_name (str): HuggingFace model identifier. Defaults to nvidia pretrained model.
        in_channels (int): Must be 3 (RGB only). Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        SegformerForSemanticSegmentation: Initialized model.

    Raises:
        ImportError: If transformers not installed.
        ValueError: If in_channels is not 3.
    """
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


def create_yolov8n(in_channels: int = 3, out_channels: int = 3):
    """
    Create YOLOv8 Nano segmentation model.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        YOLOv8SemanticSeg: Initialized nano model.
    """
    if not YOLO_SEG_AVAILABLE:
        raise ImportError("YOLOv8 segmentation not available.")
    return create_yolov8n_seg(in_channels=in_channels, out_channels=out_channels)


def create_yolov8s(in_channels: int = 3, out_channels: int = 3):
    """
    Create YOLOv8 Small segmentation model.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        YOLOv8SemanticSeg: Initialized small model.
    """
    if not YOLO_SEG_AVAILABLE:
        raise ImportError("YOLOv8 segmentation not available.")
    return create_yolov8s_seg(in_channels=in_channels, out_channels=out_channels)


def create_yolov8m(in_channels: int = 3, out_channels: int = 3):
    """
    Create YOLOv8 Medium segmentation model.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        YOLOv8SemanticSeg: Initialized medium model.
    """
    if not YOLO_SEG_AVAILABLE:
        raise ImportError("YOLOv8 segmentation not available.")
    return create_yolov8m_seg(in_channels=in_channels, out_channels=out_channels)


def create_yolov8l(in_channels: int = 3, out_channels: int = 3):
    """
    Create YOLOv8 Large segmentation model.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 3.

    Returns:
        YOLOv8SemanticSeg: Initialized large model.
    """
    if not YOLO_SEG_AVAILABLE:
        raise ImportError("YOLOv8 segmentation not available.")
    return create_yolov8l_seg(in_channels=in_channels, out_channels=out_channels)


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
    "yolov8n": create_yolov8n,
    "yolov8s": create_yolov8s,
    "yolov8m": create_yolov8m,
    "yolov8l": create_yolov8l,
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
    "yolov8n": {},
    "yolov8s": {},
    "yolov8m": {},
    "yolov8l": {},
}


def MODEL_EXPERIMENTS(filter_sets=None):
    """
    Define list of model experiments with different architectures and configurations.

    Args:
        filter_sets (list, optional): List of filter configurations to include. If None, uses default filters.

    Returns:
        list: List of tuples (model_name, mode, filters) for experimentation.
    """
    experiments = []
    for model in MODEL_BUILDERS:
        experiments.append((model, "rgb", None))
        experiments.append((model, "concat", None))
        if filter_sets is not None:
            for set_name, filters in filter_sets.items():
                experiments.append((model, "filtered", filters))

    return experiments


def build_model(name: str, **kwargs):
    """
    Factory function to build a model by name.

    Args:
        name (str): Model name (e.g., 'simple_cnn', 'unet', 'smp_unet', 'segformer', 'yolov8l').
        **kwargs: Additional arguments passed to model factory function.

    Returns:
        nn.Module: Initialized model.

    Raises:
        ValueError: If model name is not recognized.
    """
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'.")
    return MODEL_BUILDERS[name](**kwargs)


