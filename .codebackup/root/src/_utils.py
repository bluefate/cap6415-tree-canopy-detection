# From C:\github\Tree-Canopy-Detection\src\utils\config.py
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

from src.utils.helpers import c, p, t


class PathsConfig(BaseModel):
    root: Path
    train_images_zip: Optional[Path] = None
    train_images: Optional[Path] = None
    train_masks: Optional[Path] = None
    eval_images_zip: Optional[Path] = None
    eval_images: Optional[Path] = None
    eval_masks: Optional[Path] = None
    data: Optional[Path] = None
    plots: Optional[Path] = None
    annotations: Optional[Path] = None
    notebooks: Optional[Path] = None
    models: Optional[Path] = None
    checkpoint: Optional[Path] = None
    best_model: Optional[Path] = None

    @validator("*", pre = True)
    def expand_paths( cls, value, values ):
        if value is None:
            return None
        root = values.get("root", None)
        value = Path(value)
        if not value.is_absolute() and root is not None:
            return (root / value).resolve()
        return value.resolve()

class TrainConfig(BaseModel):
    image_size: int = Field(default = 256)
    batch_size: int = Field(default = 8)
    num_workers: int = Field(default = 0)
    epochs: int = Field(default = 20)
    learning_rate: float = Field(default = 1e-4)
    early_stop_patience: int = Field(default = 10)
    scheduler_factor: float = Field(default = 0.5)
    scheduler_patience: int = Field(default = 3)
    seed: int = Field(default = 42)
    best_val_loss: float = Field(default = 1e9)

def in_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


class Config(BaseModel):
    """
    Loads configuration settings from config.yaml into a typed object.
    Numeric and path values are normalized. All paths become absolute.
    """
    paths: PathsConfig
    train: TrainConfig
    extra: Dict[str, Any] = Field(default_factory = dict)


    @classmethod
    def load( cls, yaml_path: Path = None, root: Path = None ) -> "Config":

        if root is None:
            raise ValueError("Missing required field 'root'. Pass it via load(root=...).")

        if yaml_path is None:
            load_dotenv()
            project_root: Path = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
            yaml_path = project_root / "config.yaml"

        yaml_path = Path(yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing config file {yaml_path}")

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        raw_paths = raw.get("paths", { })
        raw_paths["root"] = str(root)


        raw_train = raw.get("train", { })
        extra = { k: v for k, v in raw.items() if k not in ["paths", "train"] }

        paths_cfg = PathsConfig(**raw_paths)
        train_cfg = TrainConfig(**raw_train)

        instance = cls(paths=paths_cfg, train=train_cfg, extra=extra)
        instance.auto_adjust()
        return instance

    def show( self ):
        """
        Print all configuration fields in a readable form.
        """
        t("Config settings")

        project_root = str(Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve())
        p("", f"(removed {project_root} from paths)")


        def clean_value(v):
            # None stays None
            if v is None:
                return "None"

            s = str(v)

            # remove project root prefix
            if project_root in s:
                s = s.replace(project_root, "").lstrip("/\\")

            # avoid empty result
            if s == "":
                s = "(root)"

            return s

        p("Paths")
        for k, v in self.paths.dict().items():
            p("", f"  {k}: {clean_value(v)}")

        p("Train parameters")
        for k, v in self.train.dict().items():
            p("", f"  {k}: {clean_value(v)}")

        if self.extra:
            p("Extra")
            for k, v in self.extra.items():
                p("", f"  {k}: {clean_value(v)}")


    def auto_adjust( self ):
        if self.train.image_size >= 512:
            if self.train.batch_size > 4:
                p("WARNING", f"Batch size {self.train.batch_size} too large for image size "
                             f"{self.train.image_size}. "
                             f"Reducing batch size to 4 to prevent OOM", color1 = c.RED)
                self.train.batch_size = 4

            # Also reduce workers for large images
            if self.train.num_workers > 2:
                p("WARNING", f"Num of Workers size {self.train.num_workers} too large for image "
                             f"size {self.train.image_size}. "
                             f"Reducing num_workers to 2 to prevent OOM", color1 = c.RED)
                self.train.num_workers = 2

    @property
    def MASK_COLORS(self):
        default_colors = {
            "individual_tree": [0, 255, 0],
            "group_of_trees": [255, 0, 0]
        }
        user_colors = self.extra.get("MASK_COLORS", {})
        merged = default_colors.copy()
        merged.update(user_colors)
        return merged


# From C:\github\Tree-Canopy-Detection\src\utils\helpers.py
import json
import numbers
import random
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import plotly.io as pio
import torch


# def to_chw(arr):
#     if isinstance(arr, np.ndarray):
#         return arr.transpose(2,0,1)
#     if isinstance(arr, torch.Tensor):
#         return arr.permute(2,0,1)
#     raise ValueError("Unsupported type")

# def to_hwc(arr):
#     if isinstance(arr, np.ndarray):
#         return arr.transpose(1,2,0)
#     if isinstance(arr, torch.Tensor):
#         return arr.permute(1,2,0)
#     raise ValueError("Unsupported type")

def format_time(seconds: float) -> str:
    # Break down into hours, minutes, seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hr")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if secs > 0 and hours == 0:  # only show seconds if < 1 hr
        parts.append(f"{secs} sec")

    return " ".join(parts)


def make_json_safe( obj: Any ) -> Any:
    """
    Convert an object into a JSON safe structure.
    Converts Path to string. Converts unsupported types to string as needed.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return { k: make_json_safe(v) for k, v in obj.items() }

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def normalize_type( obj: Any ) -> Any:
    """
    Normalize incoming config values to natural Python types.
    Handles numeric strings and numpy scalar types.
    """
    if obj is None:
        return None

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, str):
        s = obj.strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return obj

    if isinstance(obj, list):
        return [normalize_type(x) for x in obj]
    if isinstance(obj, dict):
        return { k: normalize_type(v) for k, v in obj.items() }

    if isinstance(obj, bytes):
        return obj.decode("utf8", errors = "ignore")

    return obj


def init_notebook( seed: int = 42 ) -> None:
    """
    Initialize common notebook settings. Sets random seeds and default renderers.
    """
    t("init_notebook")
    warnings.filterwarnings('ignore', category = UserWarning, module = 'albumentations')
    warnings.filterwarnings("ignore", category = FutureWarning)

  
    pio.renderers.default = "png"

    random.seed(seed)
    np.random.seed(seed)
    # try:
    #     import tensorflow as tf
    #
    #     tf.random.set_seed(seed)
    # except Exception:
    #     pass
    p("", "Done")


def data_loader( data: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int ):
    """
    Simple batch generator for in memory datasets.
    Expects a list of (image, mask) tensors.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        imgs = torch.stack([img for img, _ in batch])
        masks = torch.stack([mask for _, mask in batch])
        yield imgs, masks


class c(str, Enum):
    """Color definitions"""
    BLACK = "38;5;240"
    BLUE = "38;5;69"
    RED = "38;5;197"
    GREEN = "38;5;34"
    CYAN = "38;5;44"
    YELLOW = "38;5;220"
    MAGENTA = "38;5;201"
    SALMON = "38;5;216"
    ORANGE = "38;5;208"
    PURPLE = "38;5;93"

    @classmethod
    def print_colors( cls ):
        """Print all enum colors mappings"""
        for color in cls:
            print(f"\033[{color.value}m{color.name:<12} ({color.value})\033[0m")
        return ""

    @classmethod
    def print_color( self, obj ):
        """Print all enum colors mappings"""
        print(f"\033[{obj.value}m{obj.name:<12} ({obj.value})\033[0m")


    def __str__( self ):
        return str(self.value)

    __repr__ = __str__


class p:
    """Lightweight printer with clean introspection for lists, dicts, arrays, and numbers."""

    def __init__( self, obj: Any = "",
                  value: Any = None,
                  precision: int = 3,
                  show: int = 5,
                  schema: bool = False,
                  color1: c = c.GREEN,
                  color2: c = c.BLACK,
                  max_lines: int = 15,
                  bold: bool = False,
                  ):
        self.obj = obj
        self.value = value
        self.precision = precision
        self.show = show
        self.schema = schema
        self.color1 = color1
        self.color2 = color2
        self.max_lines = max_lines
        self.bold = bold

        self._print()

    def __repr__( self ):
        """Return empty string to avoid showing object representation."""
        return ""


    # ---------------------------------------------
    # Nested dict pretty-printer
    # ---------------------------------------------
    def _print_dict( self, d, indent = 2 ):
        pad = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{pad}{k}:")
                self._print_dict(v, indent = indent + 2)
            else:
                print(f"{pad}{k}: {v}")


    # ---------------------------------------------
    # Existence checks
    # ---------------------------------------------
    def _label_exists( self ):
        return self.obj is not None and str(self.obj) != ""

    def _value_exists( self ):
        v = self.value
        if v is None:
            return False
        if isinstance(v, str) and v == "":
            return False
        return True


    # -----------------------
    # Core print logic
    # -----------------------
    def _print( self ):

        try:
            label_exists = self._label_exists()
            value_exists = self._value_exists()

            # Case 1: both label and value exist
            if label_exists and value_exists:
                v = self.value

                # Numbers
                if isinstance(v, numbers.Number):
                    if float(v).is_integer():
                        text = f"{int(v)}"
                    else:
                        text = f"{float(v):.{self.precision}f}"
                    self.print_with_color(self.obj, text)
                    return

                # Dictionary
                # if isinstance(v, dict):
                #     self.print_with_color(self.obj, f"{len(v)} keys")
                #     for k, val in v.items():
                #         print(f"  {k}: {val}")
                #     return
                # Dict (supports nested)
                if isinstance(v, dict):
                    self.print_with_color(self.obj, f"{len(v)} keys")
                    self._print_dict(v, indent = 2)
                    return

                # List or tuple
                if isinstance(v, (list, tuple)):
                    if self.show > len(v):
                        self.show = len(v)
                    self.print_with_color(self.obj, f"{len(v)} items", value_color = c.BLACK)

                    for i, item in enumerate(v[: self.show]):
                        self.print_with_color(f"  {i}", item)
                    return

                # Torch tensor
                if isinstance(v, torch.Tensor):
                    shape = tuple(v.shape)
                    self.print_with_color(self.obj, f"torch tensor shape {shape}")
                    return

                # Numpy array
                if isinstance(v, np.ndarray):
                    self.print_with_color(self.obj, f"array shape {v.shape}")
                    return

                # Fallback for normal objects
                self.print_with_color(self.obj, v)
                return

            # Case 2: label exists but value is empty
            if label_exists and not value_exists:
                self.print_with_color(self.obj)
                return

            # Case 3: value exists but label is empty
            if value_exists and not label_exists:
                self.print_with_color("", self.value)
                return

            # Case 4: nothing provided
            self.print_with_color("", "")
            return

        except Exception as e:
            p.print_exception(e, self.obj, self.value)

    def _title( self, obj ):
        self.print_with_color(f"=== {obj} ===", bold = True)
        return

    # -----------------------
    # Color helper
    # -----------------------
    def print_with_color( self,
                          label = None,
                          value = None,
                          label_color: c = None,
                          value_color: c = None,
                          bold = None,
                          ):
        bold = bold or self.bold
        label_color = label_color or self.color1
        value_color = value_color or self.color2

        if (label is None or str(label) == "") and (value is None or str(value) == ""):
            #print("None", "None")
            print("")
            return

        elif label is None or str(label) == "":
            style = f"\033[{value_color.value}{';1' if bold else ''}m{value}\033[0m"
            print(style)
            return

        elif value is None or str(value) == "":
            style = f"\033[{label_color.value}{';1' if bold else ''}m{label}\033[0m"
            print(style)
            return

        else:
            if bold:
                print(f"\033[{label_color.value};1m{label}:\033[0m \033[{value_color.value}m{value}\033[0m")
            else:
                print(f"\033[{label_color.value}m{label}:\033[0m \033[{value_color.value};1m{value}\033[0m")
            return


    # -----------------------
    # Exception printer
    # -----------------------
    def print_exception( e: Exception, obj = None, val = None ):
        import traceback
        from dotenv import load_dotenv
        import os

        load_dotenv()
        project_root = os.getenv("PROJECT_ROOT", "C:/")

        temp = p("EXCEPTION", color1 = c.RED)
        temp.print_with_color("Error", str(e), c.RED, c.ORANGE, True)
        # p.print_with_color("Exception occurred", str(e), c.RED, c.ORANGE, bold = True)
        if obj is not None:
            temp.print_with_color("Object type", str(type(obj)), c.ORANGE, c.BLACK)
        if val is not None:
            temp.print_with_color("Value type", str(type(val)), c.ORANGE, c.BLACK)

        trace = traceback.format_exc()
        temp.print_with_color(f"Stack Trace\n", trace.replace(project_root, ""), c.BLUE, c.ORANGE, bold = True)


t = p()._title

#     @staticmethod
#     def _is_json( obj: object ) -> bool:
#         if not isinstance(obj, str):
#             return False
#         try:
#             json.loads(obj)
#             return True
#         except ValueError:
#             return False
#
#     @staticmethod
#     def _is_yaml( obj: object ) -> bool:
#         if not isinstance(obj, str):
#             return False
#         try:
#             yaml.safe_load(obj)
#             return True
#         except yaml.YAMLError:
#             return False
#
#     def _print( self ) -> None:
#         try:
#

#             # YAML config or dict-like
#             elif isinstance(self.obj, (dict, type(yaml.safe_load("a: 1")))):
#                 try:
#                     obj_copy = deepcopy(self.obj)
#
#                     paths = obj_copy.get("paths", { })
#                     root = Path(paths.get("root", ".")).resolve()
#
#                     p("YAML / DICT")
#
#                     # Convert Path objects to readable strings for YAML output
#                     if "paths" in obj_copy:
#                         filtered_paths = { }
#                         for key, val in obj_copy["paths"].items():
#                             val_str = str(val)
#                             root_str = str(root)
#                             val_str_norm = val_str.replace("\\", "/")
#                             root_str_norm = root_str.replace("\\", "/")
#
#                             if key == "root":
#                                 filtered_paths[key] = val_str
#                             else:
#                                 relative = val_str_norm.replace(root_str_norm, "")
#                                 if relative.startswith("/") or relative.startswith("\\"):
#                                     relative = relative[1:]
#                                 filtered_paths[key] = relative or "."
#                         obj_copy["paths"] = filtered_paths
#
#                     # Convert any remaining Path objects elsewhere to strings
#                     for key, val in obj_copy.items():
#                         if isinstance(val, dict):
#                             obj_copy[key] = {
#                                 k: str(v) if isinstance(v, Path) else v
#                                 for k, v in val.items()
#                             }
#
#                     text = yaml.dump(obj_copy, indent = 4, sort_keys = False)
#                     print(text)
#                 except Exception as e:
#                     self.print_exception(e, self.obj, self.value)
#
#                 print()
#                 return
#
#
#             # SMP Unet model
#             elif isinstance(self.obj, smp.Unet):
#                 total_params = sum(i.numel() for i in self.obj.parameters())
#                 p("Total parameters", total_params)
#                 p("Encoder")
#                 p("", self.obj.encoder, color = "black")
#                 p("Decoder")
#                 p("", self.obj.decoder, color = "black")
#                 p("Segmentation Head")
#                 p("", self.obj.segmentation_head, color = "black")
#                 return
#
#             # Pandas DataFrame
#             elif isinstance(self.obj, pd.DataFrame):
#                 p("DataFrame Preview", color = self.color2)
#                 print(self.obj.head(self.show))
#                 if self.schema:
#                     p("Number of columns", len(self.obj.columns))
#                     p("Number of rows", len(self.obj))
#                     p("Summary Stats")
#                     print(self.obj.describe())
#                     p("Schema (dtypes)", "")
#                     print(self.obj.dtypes)
#                 print()
#                 return
#

import datetime
import torch


# Assume 'p', 't', 'c', 'entries', and 'config' are defined in the context.

def format_time( seconds ):
    """Converts a total number of seconds into a human-readable D days, HH:MM:SS format."""
    td = datetime.timedelta(seconds = int(seconds))
    time_str = str(td)

    # Handle the 'days' case (e.g., "1 day, 0:03:20" -> "1d 0h 3m 20s")
    if 'day' in time_str:
        parts = time_str.split(', ')
        days = parts[0].replace(' days', 'd').replace(' day', 'd')
        hms = parts[1].split(':')
        return f"{days} {hms[0].zfill(1)}h {hms[1].zfill(2)}m {hms[2].zfill(2)}s"

    # If less than a day, output Hh Mm Ss (e.g., "3:25:45" -> "3h 25m 45s")
    hms = time_str.split(':')
    # Use lstrip('0') to show '3h' instead of '03h' unless it's '0h'
    return f"{hms[0].lstrip('0')}h {hms[1]}m {hms[2]}s"



def simple_estimate_runtime(config):
    """Estimate total runtime."""
    t("Runtime Estimate")
    from src.data.annotations import load_json_annotations

    entries = load_json_annotations(config.paths.annotations)
    train_size = int(0.8 * len(entries))

    batch_size = config.train.batch_size
    batches_per_epoch = train_size // batch_size

    # Assume ~1 second per batch (conservative)
    seconds_per_epoch = batches_per_epoch * 1

    # 4 experiments × 10 epochs
    total_seconds = 4 * 10 * seconds_per_epoch

    p("Training samples", train_size)
    p("Batches per epoch", batches_per_epoch)
    p("Estimated time per epoch", f"~{format_time(seconds_per_epoch // 60)}", color1 = c.BLACK, color2 = c.RED)
    p("Estimated total time", f"~{format_time(total_seconds)}", color1 = c.BLACK, color2 = c.RED)




def estimate_runtime_by_epcoh( experiments, epochs_per_exp = 10 ):
    """Rough estimate of total training time"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Time per epoch estimates (in seconds)
    time_per_epoch = {
        "simple_cnn": 30,
        "unet":       60,
        "yolov8s":    90,
    }

    total_seconds = 0
    for model_name, mode, _ in experiments:
        base_time = time_per_epoch.get(model_name, 60)
        # Filtered mode adds ~20% overhead
        if mode == "filtered":
            base_time *= 1.2
        total_seconds += base_time * epochs_per_exp

    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    p("Runtime Estimate", "", color1 = c.ORANGE)
    p("  Device", device.type.upper())
    p("  Experiments", len(experiments))
    p("  Epochs per exp", epochs_per_exp)
    p("  Estimated time", f"{total_hours:.1f} hours ({total_minutes:.0f} min)")

    if device.type == "cpu":
        p("  ⚠ WARNING", "CPU training is 10-20x slower!", color1 = c.RED)

    return total_hours






def estimate_runtime( experiments, config, entries = None ):
    """
    Estimate training runtime with GPU/CPU awareness, model complexity,
    and experiment mode (rgb, filtered, concat) awareness.
    """

    # --- Setup and Initialization ---
    train_size = int(0.8 * len(entries))
    batch_size = config.train.batch_size
    batches_per_epoch = max(1, train_size // batch_size)
    epochs = config.train.epochs
    total_seconds = 0.0

    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 4:  # Low-end GPU
            base_seconds = 2.0
        elif gpu_memory < 8:  # Mid-range GPU
            base_seconds = 1.0
        else:  # High-end GPU
            base_seconds = 0.5
    else:
        base_seconds = 5.0  # CPU is significantly slower

    # Mode timing multipliers (updated for clarity and to incorporate overhead)
    mode_multiplier = {
        "rgb":      1.0,  # Standard 3-channel input
        "filtered": 1.5,  # Filter computation overhead
        "concat":   2.5  # 6-channel input + filter overhead (higher than 2.0 to account for extra memory/ops)
    }

    model_complexity = {
        "simple_cnn":        1.0,
        "unet":              2.5,
        "smp_deeplabv3plus": 4.5,
        "segformer":         3.5,
        "yolov8n":           1.8,
        "yolov8l":           6.0,
    }

    # --- Header and Pre-Run Info ---
    t("Runtime Estimate")
    p("Device", device_name, color1 = c.GREEN)
    p("Total Experiments", len(experiments), color1 = c.BLACK, color2 = c.ORANGE)
    p("Training samples", train_size, color1 = c.BLACK)
    p("Batches per epoch", batches_per_epoch, color1 = c.BLACK)
    p("Epochs per experiment", epochs, color1 = c.BLACK)
    p("Batch size", batch_size, color1 = c.BLACK)
    p("-" * 70, color1 = c.ORANGE)
    p()

    p("Per-Experiment Estimates:", color1 = c.CYAN, bold = True)
    p("-" * 70, color1 = c.CYAN)

    # --- Calculation Loop ---
    for model_name, mode, filters in experiments:

        # Get multipliers, defaulting to 1.0 if model/mode not found
        mode_mult = mode_multiplier.get(mode, 1.0)
        complexity_mult = model_complexity.get(model_name, 1.0)

        # Total multiplier
        total_mult = mode_mult * complexity_mult

        # Time calculation: Base * Device/Complexity Multipliers * (Batches * Epochs)
        sec_per_batch = base_seconds * total_mult

        exp_seconds = epochs * batches_per_epoch * sec_per_batch
        total_seconds += exp_seconds

        # Format filter string
        filter_str = f"[{', '.join(filters)}...]" if filters and len(filters) > 0 else "none"

        # Display
        exp_label = f"{model_name:20s} | {mode:8s} | {filter_str:20s}"

        p(exp_label, format_time(exp_seconds), color1 = c.BLUE, color2 = c.BLACK)

    p()
    t("Totals")

    total_hours = total_seconds / 3600

    p("Total experiments", len(experiments))
    p("Total batches", batches_per_epoch * epochs * len(experiments))

    p("Estimated total time", format_time(total_seconds), color1 = c.GREEN, bold = True)

    # Time breakdown and Warnings
    if total_hours >= 24:
        p("Estimated completion", f"~{total_hours / 24:.1f} days", color1 = c.ORANGE)
        p("⚠ WARNING", "Training will take over 24 hours!", color1 = c.ORANGE, bold = True)
        p("Consider", "Reducing epochs or selecting fewer models", color1 = c.ORANGE)
    elif total_hours >= 8:
        p("Estimated completion", f"~{total_hours:.1f} hours", color1 = c.ORANGE)
        p("⚠ NOTE", "Long training session - consider running overnight", color1 = c.ORANGE)
    else:
        p("Estimated completion", f"~{total_seconds / 60:.0f} minutes", color1 = c.GREEN)

    if device.type == 'cpu':
        p("⚠ CPU DETECTED", "Training on CPU is 10-20x slower than GPU", color1 = c.RED, bold = True)
    p()




# From C:\github\Tree-Canopy-Detection\src\utils\image_converter.py

"""
Image format conversion utilities for conversion from TIFF to PNG.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from src.utils.helpers import c, p
from src.utils.logging import Logger


class ImageConverter:
    """
    Convert TIFF images to PNG format for reliable deep learning training.

    TIFF files, especially GeoTIFFs from aerial imagery, can cause:
    - Memory issues during loading
    - Inconsistent library support
    - System crashes during training

    PNG provides:
    - Lossless compression
    - Universal library support
    - Faster loading during training
    - Stable memory usage
    """

    def __init__(self, source_dir: Path, target_dir: Optional[Path] = None, logger: Optional[Logger] = None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir) if target_dir else self.source_dir
        self.logger = logger or Logger()

        self.target_dir.mkdir(parents=True, exist_ok=True)

    def find_tiff_files(self) -> List[Path]:
        """Find all TIFF files in source directory."""
        tiff_files = []
        for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tiff_files.extend(self.source_dir.glob(pattern))
        return sorted(tiff_files)

    def convert_single(self, tiff_path: Path, quality: int = 95) -> Tuple[bool, Optional[str]]:
        """
        Convert a single TIFF file to PNG.
        """
        try:
            # Open TIFF with PIL (better TIFF support than OpenCV)
            img = Image.open(tiff_path)

            # Convert to RGB if needed
            if img.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Generate output path
            png_path = self.target_dir / tiff_path.with_suffix('.png').name

            # Save as PNG with optimization
            img.save(png_path, 'PNG', optimize=True, compress_level=9)

            return True, None

        except Exception as e:
            return False, str(e)

    def convert_batch(self, overwrite: bool = False) -> dict:
        """
        Convert all TIFF files to PNG.
        """
        tiff_files = self.find_tiff_files()

        if not tiff_files:
            self.logger.warn("No TIFF files found in source directory")
            return {"total": 0, "converted": 0, "skipped": 0, "failed": 0}

        self.logger.info(f"Found {len(tiff_files)} TIFF files")

        stats = {
            "total": len(tiff_files),
            "converted": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }

        for tiff_path in tqdm(tiff_files, desc="Converting TIFFs to PNG"):
            png_path = self.target_dir / tiff_path.with_suffix('.png').name

            # Skip if PNG already exists and overwrite is False
            if png_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            success, error = self.convert_single(tiff_path)

            if success:
                stats["converted"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append({
                    "file": tiff_path.name,
                    "error": error
                })
                self.logger.error(f"Failed to convert {tiff_path.name}: {error}")

        # Print summary
        self.logger.header("Conversion Summary")
        self.logger.info(f"Total files: {stats['total']}")
        self.logger.info(f"Converted: {stats['converted']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        self.logger.info(f"Failed: {stats['failed']}")

        if stats["errors"]:
            self.logger.warn(f"Errors occurred in {len(stats['errors'])} files")
            for err in stats["errors"][:5]:  # Show first 5 errors
                self.logger.error(f"  {err['file']}: {err['error']}")

        return stats

    def update_annotations(self, annotations_path: Path, output_path: Optional[Path] = None, create_backup: bool = True):
        """
        Update annotation file to reference PNG files instead of TIFF.
        """
        import shutil
        from datetime import datetime

        annotations_path = Path(annotations_path)
        output_path = Path(output_path) if output_path else annotations_path

        # Create backup before modifying
        if create_backup and output_path == annotations_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = annotations_path.with_suffix(f'.json.backup_{timestamp}')

            # Also create a simple .backup without timestamp (latest backup)
            simple_backup = annotations_path.with_suffix('.json.backup')

            try:
                shutil.copy2(annotations_path, backup_path)
                shutil.copy2(annotations_path, simple_backup)
                self.logger.info(f"Created backup: {backup_path.name}")
                self.logger.info(f"Latest backup: {simple_backup.name}")
            except Exception as e:
                self.logger.error(f"Failed to create backup: {e}")
                raise RuntimeError(f"Cannot modify annotations without backup: {e}")

        # Load original annotations
        with open(annotations_path, 'r', encoding='utf8') as f:
            data = json.load(f)

        # Update file references
        updated_count = 0
        for img_entry in data.get("images", []):
            filename = img_entry.get("file_name", "")
            if filename.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                # Change extension to .png
                new_filename = Path(filename).with_suffix('.png').name
                img_entry["file_name"] = new_filename
                updated_count += 1

        # Write updated annotations
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4)

        self.logger.info(f"Updated {updated_count} file references in annotations")
        return updated_count

    @staticmethod
    def restore_annotations_from_backup(annotations_path: Path, backup_name: Optional[str] = None) -> bool:
        """
        Restore annotations from backup.
        """
        import shutil

        annotations_path = Path(annotations_path)

        if backup_name:
            backup_path = annotations_path.parent / backup_name
        else:
            # Use latest .backup file
            backup_path = annotations_path.with_suffix('.json.backup')

        if not backup_path.exists():
            p("Backup not found", str(backup_path))

            # List available backups
            backup_dir = annotations_path.parent
            backups = list(backup_dir.glob(f"{annotations_path.stem}.json.backup*"))

            if backups:
                p("Available backups")
                for b in sorted(backups):
                    p("", f"  {b.name}")

            return False

        try:
            shutil.copy2(backup_path, annotations_path)
            p("Restored annotations from", backup_path.name, color1= c.GREEN)

            return True
        except Exception as e:
            p("Failed to restore backup", str(e), color1=c.RED, color2 = c.RED)
            return False


def batch_convert_tiff_to_png(
    image_dir: Path,
    annotations_path: Optional[Path] = None,
    overwrite: bool = False
) -> dict:
    """
    Convenience function to convert all TIFFs and update annotations.
    """
    converter = ImageConverter(image_dir)
    stats = converter.convert_batch(overwrite=overwrite)

    if annotations_path and Path(annotations_path).exists():
        converter.update_annotations(annotations_path)

    return stats

# From C:\github\Tree-Canopy-Detection\src\utils\logging.py
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.helpers import c, p, t


class Logger:
    """
    Simple logger for console and optional file output.
    Designed for training loops, evaluation runs, and debugging.
    """

    def __init__( self, log_file: Optional[Path] = None ,cfg=None):
        self.log_file = Path(log_file) if log_file is not None else None
        self.cfg = cfg
        if self.log_file:
            self.log_file.parent.mkdir(parents = True, exist_ok = True)

    def write( self, text: str ) -> None:
        """
        Write a line of text to console and file if configured.
        """
        timestamp = datetime.now().strftime("%m.%d %H:%M:%S")
        line = f"[{timestamp}] {text}"

        if self.log_file is not None:
            with open(self.log_file, "a", encoding = "utf8") as f:
                f.write(line + "\n")

    def header( self, text: str ) -> None:
        """
        Write a visible section header.
        """
        self.write(f"=== {text} ===")
        t(text)

    # def info( self, text: str ) -> None:
    #     """
    #     Write an informational line.
    #     """
    #     self.write(text)
    #     p("[Info]", text, color1 = c.BLUE, color2 = c.BLACK)

    def info(self, text):
        """
        Write informational output. If 'text' is a PyTorch model,
        display a clean torchinfo summary instead of the raw model dump.
        """
        from torchinfo import summary
        import torch.nn as nn

        # If the user passes a model
        if isinstance(text, nn.Module):
            try:
                # Try to infer model input channels from common cases
                in_channels = 3
                if hasattr(text, 'in_channels'):
                    in_channels = text.in_channels

                # Default spatial size (can be adjusted)
                img_size = self.cfg.train.image_size

                model_summary = summary(
                        text,
                        input_size=(1, in_channels, img_size, img_size),
                        depth=3,
                        col_names=("input_size", "output_size", "num_params")
                )

                self.write(str(model_summary))
                p("", str(model_summary))
                # p("[Model]", "Summary printed via torchinfo", color1=c.BLUE, color2=c.BLACK)
                return

            except Exception as e:
                # Fallback to normal printing if summary fails
                self.write(str(text))
                p("[Info]", f"torchinfo failed: {e}", color1=c.ORANGE)
                return

        # Normal text logging
        self.write(str(text))
        p("[Info]", str(text), color1=c.BLUE, color2=c.BLACK)


    def warn( self, text: str ) -> None:
        """
        Write a warning line.
        """
        self.write(f"Warning: {text}")
        p("[Warn]", text, color1 = c.ORANGE, color2 = c.BLACK)

    def error( self, text: str ) -> None:
        """
        Write an error line.
        """
        self.write(f"Error: {text}")
        p("[Error]", text, color1 = c.RED, color2 = c.BLACK)

    def exception(self, text: str, exc: Exception) -> None:
        """
        Log an exception with traceback.
        """
        import traceback

        error_msg = f"Exception: {text}\n{str(exc)}\n{traceback.format_exc()}"
        self.write(error_msg)
        p("[Exception]", text, color1=c.RED, color2=c.RED)
        p("", str(exc), color1=c.ORANGE, color2=c.ORANGE)

# From C:\github\Tree-Canopy-Detection\src\utils\preflight.py
import shutil
import traceback
from pathlib import Path

import cv2
import torch

from data.annotations import load_json_annotations
from data.augmentations import get_val_augmentations
from data.loaders import ImageMaskDataset
from models.zoo import build_model
from src.utils.helpers import c, p, t


def check_config(config):
    """Verify config loads correctly."""
    t("Checking Configuration")
    try:
        p("✓ Config loaded", config.paths.root, color1 = c.GREEN)

        # Check critical paths
        checks = [
            ('annotations', config.paths.annotations),
            ('train_images', config.paths.train_images),
            ('eval_images', config.paths.eval_images),
            ('models', config.paths.models),
            ('notebooks', config.paths.notebooks),
        ]

        for name, path in checks:
            if path and Path(path).exists():
                p(f"✓ {name}", "exists", color1 = c.GREEN)
            else:
                p(f"✗ {name}", f"missing: {path}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Config failed", str(e), color1 = c.RED)
        return False


def check_gpu():
    """Check GPU availability."""
    t("Checking GPU")

    if torch.cuda.is_available():
        p("✓ CUDA available", torch.cuda.get_device_name(0), color1 = c.GREEN)
        p("GPU count", torch.cuda.device_count())

        # Check memory
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        p("GPU memory", f"{total:.1f} GB")

        # Test allocation
        try:
            test = torch.zeros((1000, 1000)).cuda()
            del test
            torch.cuda.empty_cache()
            p("✓ GPU allocation", "working", color1 = c.GREEN)
        except Exception as e:
            p("✗ GPU allocation", str(e), color1 = c.RED)

        return True
    else:
        p("✗ CUDA not available", "will use CPU (slow)", color1 = c.ORANGE)
        return False


def check_data(config):
    """Check data can be loaded."""
    t("Checking Data")

    try:
        entries = load_json_annotations(config.paths.annotations)

        p("✓ Annotations loaded", f"{len(entries)} images", color1 = c.GREEN)

        # Check first entry
        entry = entries[0]
        img_path = config.paths.train_images / entry.image_path.name

        if img_path.exists():
            p("✓ Sample image", "found", color1 = c.GREEN)

            img = cv2.imread(str(img_path))
            if img is not None:
                p("✓ Image loading", f"shape={img.shape}", color1 = c.GREEN)
            else:
                p("✗ Image loading", "failed", color1 = c.RED)
        else:
            p("✗ Sample image", f"not found: {img_path}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Data check failed", str(e), color1 = c.RED)
        return False


def check_dataset(config):
    """Test dataset creation."""
    t("Checking Dataset")

    try:
        entries = load_json_annotations(config.paths.annotations)
        transform = get_val_augmentations(config.train.image_size)

        dataset = ImageMaskDataset(
                entries[:5],
                config.paths.train_images,
                transform = transform
        )

        p("✓ Dataset created", f"{len(dataset)} samples", color1 = c.GREEN)

        # Test loading
        img_t, mask_t = dataset[0]

        p("✓ Image shape", img_t.shape, color1 = c.GREEN)
        p("✓ Mask shape", mask_t.shape, color1 = c.GREEN)

        # Verify shapes
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            p("✓ Image format", "correct [3, H, W]", color1 = c.GREEN)
        else:
            p("✗ Image format", f"wrong: {img_t.shape}", color1 = c.RED)

        # Multi-class segmentation: masks should be [H, W] with class indices (0, 1, 2)
        if mask_t.ndim == 2:
            p("✓ Mask format", f"correct [H, W] for multi-class", color1 = c.GREEN)
            unique_vals = torch.unique(mask_t)
            if torch.all((unique_vals >= 0) & (unique_vals <= 2)):
                p("✓ Mask values", f"valid classes: {unique_vals.tolist()}", color1 = c.GREEN)
            else:
                p("✗ Mask values", f"invalid: {unique_vals.tolist()}", color1 = c.RED)
        elif mask_t.ndim == 3 and mask_t.shape[0] == 1:
            p("⚠ Mask format", "[1, H, W] - should be [H, W] for CrossEntropyLoss", color1 = c.ORANGE)
        else:
            p("✗ Mask format", f"wrong: {mask_t.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Dataset check failed", str(e), color1 = c.RED)

        traceback.print_exc()
        return False


def check_model():
    """Test model creation."""
    t("Checking Model")

    try:
        model = build_model('simple_cnn', in_channels = 3, out_channels = 3)

        p("✓ Model created", "simple_cnn", color1 = c.GREEN)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        p("Model parameters", f"{total_params:,}")

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        test_input = torch.randn(2, 3, 256, 256).to(device)

        with torch.no_grad():
            output = model(test_input)

        p("✓ Forward pass", f"output shape={output.shape}", color1 = c.GREEN)

        # Multi-class segmentation: 3 channels (background, individual_tree, group_of_trees)
        expected_shape = (2, 3, 256, 256)

        if output.shape == expected_shape:
            p("✓ Output shape", "correct (multi-class)", color1 = c.GREEN)
        else:
            p("✗ Output shape", f"expected {expected_shape}, got {output.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Model check failed", str(e), color1 = c.RED)

        traceback.print_exc()
        return False


def check_disk_space(config):
    """Check available disk space."""
    t("Checking Disk Space")

    try:
        total, used, free = shutil.disk_usage(config.paths.root)

        free_gb = free / (1024 ** 3)

        p("Free space", f"{free_gb:.1f} GB")

        if free_gb > 10:
            p("✓ Sufficient space", ">10 GB available", color1 = c.GREEN)
            return True
        elif free_gb > 5:
            p("⚠ Limited space", f"{free_gb:.1f} GB (needs >10 GB)", color1 = c.ORANGE)
            return True
        else:
            p("✗ Insufficient space", f"{free_gb:.1f} GB (needs >10 GB)", color1 = c.RED)
            return False

    except Exception as e:
        p("✗ Disk check failed", str(e), color1 = c.ORANGE)
        return True








# From C:\github\Tree-Canopy-Detection\src\utils\tester.py
import numpy as np

from src.utils.helpers import c, p


class Tester:
    """
    Utility class to quickly inspect values during development.
    """

    def __init__( self, obj = "", value = None, precision = 3, color1 = c.GREEN, color2 = c.BLACK, bold = False ):
        self.obj = obj
        self.value = value
        self.precision = precision
        self.color1 = color1
        self.color2 = color2
        self.bold = bold

    def _print( self ):
        p(self.obj,
          self.value,
          precision = self.precision,
          color1 = self.color1,
          color2 = self.color2,
          bold = self.bold,
          )


def p_test():
    p("--- Testing Number ---")
    Tester("num", 42)._print()
    Tester("float", 3.14159)._print()

    p("--- Testing Dict ---")
    Tester("dict", { "a": 1, "b": 2 })._print()

    p("--- Testing List ---")
    Tester("list", [10, 20, 30, 40, 50, 60])._print()

    p("--- Testing Tuple ---")
    Tester("tuple", ("x", "y", "z"))._print()

    p("--- Testing Numpy Array ---")
    Tester("array", np.zeros((2, 3)))._print()

    p("--- Testing String ---")
    Tester("string", "hello")._print()

    p("--- Testing Empty ---")
    Tester("", None)._print()

    p("--- Testing Color ---")
    color1 = c.ORANGE
    color2 = c.PURPLE
    c.print_color(color1)
    c.print_color(color2)
    p(f"OBJ color test {color1.name}", f"VALUE color test {color2.name}", color1 = color1, color2 = color2)

    p("--- Testing Title w/blue ---")
    Tester("Title", color1 = c.BLUE)._print()

    p("--- Testing Value w/yellow ---")
    Tester("", "Value", color2 = c.YELLOW)._print()

    p("--- Testing Bold  ---")
    Tester("Bold", "Not bold")._print()


# From C:\github\Tree-Canopy-Detection\src\utils\versioning.py
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class VersionManager:
    """
    Handle automatic version numbering for training runs.
    Creates folders that store config snapshots, model weights, metrics, and logs.
    Versioning is based on a hash of the configuration object.
    """

    def __init__( self, base_dir: Path ):
        """
        base_dir is the parent directory that holds version folders.
        For example checkpoints or runs.
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents = True, exist_ok = True)


    def _normalize_for_json( self, obj ):
        if isinstance(obj, dict):
            return { k: self._normalize_for_json(v) for k, v in obj.items() }
        if isinstance(obj, list):
            return [self._normalize_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._normalize_for_json(v) for v in obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def compute_hash( self, cfg: Dict[str, Any] ) -> str:
        """
        Compute a stable hash of the configuration dictionary.
        """
        clean = self._normalize_for_json(cfg)
        cfg_json = json.dumps(clean, sort_keys = True)
        return hashlib.md5(cfg_json.encode("utf8")).hexdigest()

    def find_latest( self ) -> Optional[Path]:
        """
        Return the path of the latest version folder or None.
        """
        versions = sorted(self.base_dir.glob("v*"))
        if not versions:
            return None
        return versions[-1]

    def create_next_version( self ) -> Path:
        """
        Create the next version folder based on existing folders.
        For example v001 then v002.
        """
        latest = self.find_latest()
        if latest is None:
            name = "v001"
        else:
            num = int(latest.name[1:])
            name = f"v{num + 1:03d}"

        target = self.base_dir / name
        target.mkdir(parents = True, exist_ok = True)
        return target

    def resolve_version( self, cfg: Dict[str, Any] ) -> Path:
        """
        Determine the correct version folder for this config.
        If the newest version has a matching hash, reuse it.
        Otherwise, create a new version.
        """
        cfg_hash = self.compute_hash(cfg)
        latest = self.find_latest()

        if latest is not None:
            meta_path = latest / "config_snapshot.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding = "utf8") as f:
                    data = json.load(f)
                saved_hash = data.get("hash", None)
                if saved_hash == cfg_hash:
                    return latest

        version_dir = self.create_next_version()
        self.save_snapshot(version_dir, cfg, cfg_hash)
        return version_dir

    def save_snapshot( self, version_dir: Path, cfg: Dict[str, Any], cfg_hash: str ) -> None:
        """
        Save configuration and hash for reproducibility.
        """
        clean_cfg = self._normalize_for_json(cfg)

        path = version_dir / "config_snapshot.json"
        with open(path, "w", encoding = "utf8") as f:
            json.dump(
                    {
                        "config": clean_cfg,
                        "hash": cfg_hash,
                        "created": datetime.now().isoformat(),
                    },
                    f,
                    indent = 2,
            )

    def get_paths( self, version_dir: Path ) -> Dict[str, Path]:
        """
        Produce standard file paths used by training and evaluation.
        """
        return {
            "checkpoint": version_dir / "checkpoint.pth",
            "best": version_dir / "best_model.pth",
            "metrics": version_dir / "metrics.csv",
            "log": version_dir / "run.log",
        }


# From C:\github\Tree-Canopy-Detection\src\utils\__init__.py


