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
    warnings.filterwarnings("ignore", category = UserWarning)
    warnings.filterwarnings("ignore", category = FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

    pio.renderers.default = "png"

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
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
                    self.print_with_color(self.obj, f"{len(v)} items")
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
            self.print_with_color("None", "None")

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
            print("None", "None")

        elif label is None or str(label) == "":
            style = f"\033[{value_color.value}{';1' if bold else ''}m{value}\033[0m"
            print(style)

        elif value is None or str(value) == "":
            style = f"\033[{label_color.value}{';1' if bold else ''}m{label}\033[0m"
            print(style)

        else:
            if bold:
                print(f"\033[{label_color.value};1m{label}:\033[0m \033[{value_color.value}m{value}\033[0m")
            else:
                print(f"\033[{label_color.value}m{label}:\033[0m \033[{value_color.value};1m{value}\033[0m")


    # -----------------------
    # Exception printer
    # -----------------------
    @staticmethod
    def print_exception( self, e: Exception, obj = None, val = None ):
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
