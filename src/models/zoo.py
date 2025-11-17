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


def create_simple_cnn( in_channels: int = 3, out_channels: int = 1 ):
    return SimpleCNN(in_channels = in_channels, out_channels = out_channels)


def create_unet( in_channels: int = 3, out_channels: int = 1 ):
    return UNet(in_channels = in_channels, out_channels = out_channels)


def create_smp_unet(
        encoder_name = "resnet34",
        encoder_weights = "imagenet",
        in_channels = 3,
        out_channels = 1,
):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.Unet(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = out_channels,
    )


def create_smp_fpn( encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 3, out_channels = 1 ):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.FPN(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = out_channels,
    )


def create_smp_linknet( encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 3, out_channels = 1 ):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.Linknet(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = outchannels,
    )


def create_smp_deeplabv3( encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 3, out_channels = 1 ):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.DeepLabV3(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = out_channels,
    )


def create_smp_deeplabv3plus( encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 3,
                              out_channels = 1 ):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is not installed")
    return smp.DeepLabV3Plus(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = out_channels,
    )


def create_segformer( model_name = "nvidia/segformer-b0-finetuned-ade-512-512", out_channels = 1 ):
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed")
    model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels = out_channels,
    )
    return model


def create_timm_segformer( encoder_name = "tf_efficientnetv2_s", out_channels = 1 ):
    if not TIMM_AVAILABLE:
        raise ImportError("timm not installed")
    backbone = timm.create_model(
            encoder_name,
            features_only = True,
            pretrained = True,
            in_chans = 3,
    )
    raise NotImplementedError("timm segformer head integration needs custom head")


def create_timm_upernet( encoder_name = "swin_base_patch4_window7_224", out_channels = 1 ):
    if not TIMM_AVAILABLE:
        raise ImportError("timm not installed")
    raise NotImplementedError("UPerNet using timm backbone available if needed. Ask to enable.")


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
    "simple_cnn": { },
    "unet": { },
    "smp_unet": { "encoder_name": "resnet34" },
    "smp_fpn": { "encoder_name": "resnet34" },
    "smp_linknet": { "encoder_name": "resnet34" },
    "smp_deeplabv3": { "encoder_name": "resnet34" },
    "smp_deeplabv3plus": { "encoder_name": "resnet34" },
    "segformer": { "model_name": "nvidia/segformer-b0-finetuned-ade-512-512" },
}


def build_model( name: str, **kwargs ):
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name}")
    return MODEL_BUILDERS[name](**kwargs)
