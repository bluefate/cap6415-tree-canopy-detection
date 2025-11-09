# +
# imports
import json
import numbers
import random as r
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.io as pio
import segmentation_models_pytorch as smp
import torch
import yaml


# Helper functions (keep outside the class)
def format_number( n ):
    if n == 0:
        return "0"
    elif n < 1:
        return f"{n:.1e}"
    elif n >= 1_000_000:  # 1e6
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:  # 1e3
        return f"{n / 1_000:.2f}K"
    else:
        return f"{n}"


def init_this_notebook( seed = 42 ):
    p("Python", sys.version)
    p("Numpy", np.__version__)
    p("Panda", pd.__version__)
    p("Torch", torch.__version__)

    # ignore warnings
    warnings.filterwarnings("ignore", category = UserWarning)
    warnings.filterwarnings("ignore", category = FutureWarning)

    warnings.filterwarnings("ignore", category = UserWarning, module = "tensorflow")

    # set default plot renderer to png
    pio.renderers.default = "png"

    r.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    p()


def normalize_paths( config: dict ) -> dict:
    """Convert all paths in config['paths'] to Path objects, resolving relative to root."""
    paths = config.get("paths", { })
    root = Path(paths.get("root", ".")).resolve()

    normalized = { }
    for key, val in paths.items():
        path = val if isinstance(val, Path) else Path(val)

        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()

        normalized[key] = path

    if "root" not in normalized:
        normalized["root"] = root

    config["paths"] = normalized
    return config


def normalize_type( obj: Any ) -> Any:
    """Return obj as its most natural Python type.
    Handles ints, floats (including scientific notation),
    strings, bools, lists, dicts, numpy types, and paths.
    """
    if obj is None:
        return None

    # Pass-through for primitives
    if isinstance(obj, (int, float, bool)):
        return obj

    # Convert NumPy scalar types to native
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except ImportError:
        pass

    # Convert strings that look numeric
    if isinstance(obj, str):
        s = obj.strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass
        # Try float (covers scientific notation)
        try:
            return float(s)
        except ValueError:
            return s  # not numeric, return as string

    # Recursively handle containers
    if isinstance(obj, list):
        return [normalize_type(x) for x in obj]

    if isinstance(obj, dict):
        return { k: normalize_type(v) for k, v in obj.items() }

    # Convert Path and bytes
    if isinstance(obj, Path):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors = "ignore")

    # Everything else stays as-is
    return obj


def data_loader( data, batch_size ):
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        images = torch.stack([img for img, _ in batch])
        masks = torch.stack([mask for _, mask in batch])
        yield images, masks


# p class
class p:
    """Pretty printer with color support and format detection."""

    color_codes = {
        "black": "38;5;240",
        "blue": "38;5;69",
        "red": "38;5;197",
        "green": "38;5;34",
        "cyan": "38;5;44",
        "yellow": "38;5;220",
        "salmon": "38;5;5",
    }

    def __init__(
            self,
            obj: Any = "",
            value: Optional[Any] = None,
            precision: int = 3,
            show: int = 3,
            schema: bool = False,
            color: str = "green",
            color2: str = "black",
    ):
        self.obj = obj
        self.value = value
        self.precision = precision
        self.show = show
        self.schema = schema
        self.color = self.color_codes.get(color.lower(), color)
        self.color2 = self.color_codes.get(color2.lower(), color2)

        self._print()

    def __repr__( self ):
        """Return empty string to avoid showing object representation."""
        return ""

    @staticmethod
    def _is_json( obj: object ) -> bool:
        if not isinstance(obj, str):
            return False
        try:
            json.loads(obj)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_yaml( obj: object ) -> bool:
        if not isinstance(obj, str):
            return False
        try:
            yaml.safe_load(obj)
            return True
        except yaml.YAMLError:
            return False

    def _print( self ) -> None:
        try:
            # List handling
            if isinstance(self.obj, list):
                p(f"List (first {self.show} items):", color = self.color2)
                for i, item in enumerate(self.obj[:self.show]):
                    p(f"{i}", item)
                if self.schema:
                    p("List length", len(self.obj))
                    p("Contained types", set(type(x) for x in self.obj))
                print()
                return

            # JSON string
            elif self._is_json(self.obj) and self._is_yaml(self.obj):
                try:
                    p("JSON")
                    parsed_data = json.loads(self.obj)
                    pretty_json = json.dumps(parsed_data, indent = 4, sort_keys = True)
                    print(pretty_json)
                    return
                except Exception as e:
                    p(f"Failed to print JSON: {e}")
                print()
                return

            # YAML config or dict-like
            elif isinstance(self.obj, (dict, type(yaml.safe_load("a: 1")))):
                try:
                    cfg = deepcopy(self.obj)

                    paths = cfg.get("paths", { })
                    root = Path(paths.get("root", ".")).resolve()

                    p("YAML / DICT")

                    # Convert Path objects to readable strings for YAML output
                    if "paths" in cfg:
                        filtered_paths = { }
                        for key, val in cfg["paths"].items():
                            val_str = str(val)
                            root_str = str(root)
                            val_str_norm = val_str.replace("\\", "/")
                            root_str_norm = root_str.replace("\\", "/")

                            if key == "root":
                                filtered_paths[key] = val_str
                            else:
                                relative = val_str_norm.replace(root_str_norm, "")
                                if relative.startswith("/") or relative.startswith("\\"):
                                    relative = relative[1:]
                                filtered_paths[key] = relative or "."
                        cfg["paths"] = filtered_paths

                    # Convert any remaining Path objects elsewhere to strings
                    for key, val in cfg.items():
                        if isinstance(val, dict):
                            cfg[key] = {
                                k: str(v) if isinstance(v, Path) else v
                                for k, v in val.items()
                            }

                    text = yaml.dump(cfg, indent = 4, sort_keys = False)
                    print(text)
                except Exception as e:
                    p(f"Failed to print YAML: {e}")

                print()
                return

            # NumPy arrays
            elif isinstance(self.obj, np.ndarray):
                p(f"NumPy array shape: {self.obj.shape}", color = self.color2)
                p("Preview", self.obj[:self.show])
                if self.schema:
                    p("Array shape", self.obj.shape)
                    p("Array dtype", self.obj.dtype)
                print()
                return

            # SMP Unet model
            elif isinstance(self.obj, smp.Unet):
                total_params = sum(i.numel() for i in self.obj.parameters())
                p("Total parameters", total_params)
                p("Encoder")
                p("", self.obj.encoder, color = "black")
                p("Decoder")
                p("", self.obj.decoder, color = "black")
                p("Segmentation Head")
                p("", self.obj.segmentation_head, color = "black")
                return

            # Pandas DataFrame
            elif isinstance(self.obj, pd.DataFrame):
                p("DataFrame Preview", color = self.color2)
                print(self.obj.head(self.show))
                if self.schema:
                    p("Number of columns", len(self.obj.columns))
                    p("Number of rows", len(self.obj))
                    p("Summary Stats")
                    print(self.obj.describe())
                    p("Schema (dtypes):", "")
                    print(self.obj.dtypes)
                print()
                return

            # Generic value printing
            if self.value is not None:
                if isinstance(self.value, numbers.Number):
                    val = float(self.value)
                    if float(val).is_integer():
                        formatted = f"{int(val):,}"
                    else:
                        formatted = f"{val:,.{self.precision}f}"

                    if str(self.obj) == "":
                        print(f"\033[{self.color}m{formatted}\033[0m")
                    else:
                        print(
                            f"\033[{self.color}m{self.obj}:\033[0m \033[{self.color2};1m{formatted}\033[0m",
                        )
                else:
                    if str(self.obj) == "":
                        print(f"\033[{self.color}m{self.value}\033[0m")
                    else:
                        print(
                            f"\033[{self.color};1m{self.obj}:\033[0m \033[{self.color2}m{self.value}\033[0m",
                        )

            elif self.obj:
                print(f"\033[{self.color}m\033[1m\n=== {self.obj} ===\033[0m")
            else:
                print()

        except Exception as e:
            print("Exception occurred", str(e))
            print("Type of object passed", type(self.obj))
            raise

# Backward compatibility - allows p() function calls
# p = P
