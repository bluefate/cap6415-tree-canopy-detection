# +
# imports
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numbers
import numpy as np
import pandas as pd
import plotly.io as pio
import random as r
import sys
import warnings
import yaml

# import tensorflow as tf


color_codes = {
    # "reset": "0",
    "black": "38;5;240",
    "blue": "38;5;69",
    "red": "38;5;197",
    "green": "38;5;34",
    "cyan": "38;5;44",
    "yellow": "38;5;220",
    "salmon": "38;5;5",
}


def p(
    obj: Any = "",
    value: Optional[Any] = None,
    precision: int = 3,
    toHTML: bool = False,
    show: int = 3,
    showIndex: bool = False,
    schema: bool = False,
    color: str = "green",
    color2: str = "black",
) -> None:
    _color = color_codes.get(color.lower(), color)
    _color2 = color_codes.get(color2.lower(), color2)

    try:
        if isinstance(obj, list):
            p(f"List (first {show} items):", color=color2)
            for i, item in enumerate(obj[:show]):
                p(f"{i}", item)
            if schema:
                p("List length", len(obj))
                p("Contained types", set(type(x) for x in obj))
            p()
            return

        # YAML config or dict-like
        elif isinstance(obj, (dict, type(yaml.safe_load("a: 1")))):
            try:
                cfg = deepcopy(obj)

                paths = cfg.get("paths", {})
                root = Path(paths.get("root", ".")).resolve()

                p("YAML Configuration")

                # Convert Path objects to readable strings for YAML output
                if "paths" in cfg:
                    filtered_paths = {}
                    for key, val in cfg["paths"].items():
                        val_str = str(val)
                        root_str = str(root)
                        # Normalize slashes for consistent replacement
                        val_str_norm = val_str.replace("\\", "/")
                        root_str_norm = root_str.replace("\\", "/")

                        if key == "root":
                            filtered_paths[key] = val_str
                        else:
                            # Remove the root part for brevity
                            relative = val_str_norm.replace(root_str_norm, "")
                            # Clean up leading slashes
                            if relative.startswith("/") or relative.startswith("\\"):
                                relative = relative[1:]
                            filtered_paths[key] = relative or "."
                    cfg["paths"] = filtered_paths

                # Convert any remaining Path objects elsewhere to strings
                for key, val in cfg.items():
                    if isinstance(val, dict):
                        cfg[key] = {k: str(v) if isinstance(v, Path) else v for k, v in val.items()}

                text = yaml.dump(cfg, sort_keys=False)
                #p(text)
                for i, line in enumerate(text.splitlines(), start=1):
                    p(f"{i:02d}", line)


            except Exception as e:
                p(f"Failed to print YAML: {e}")

            p()
            return

        # NumPy arrays
        elif isinstance(obj, np.ndarray):
            p(f"NumPy array shape: {obj.shape}", color=color2)
            p("Preview", obj[:show])
            if schema:
                p("Array shape", obj.shape)
                p("Array dtype", obj.dtype)
            p()
            return

        # TensorFlow Dataset
        # elif isinstance(obj, tf.data.Dataset):
        #     # forcing actual count of items in obj
        #     count = sum(1 for _ in obj)
        #     p("Actual element count", count, color="blue", color2="red")

        #     if schema:
        #         sample = next(iter(obj.take(1)))
        #         if isinstance(sample, tuple):
        #             for i, s in enumerate(sample):
        #                 p(f"Sample {i}", s.numpy()[:2] if hasattr(s, "numpy") else s)
        #         else:
        #             p(
        #                 "Sample Element",
        #                 sample.numpy()[:2] if hasattr(sample, "numpy") else sample,
        #             )

        #     p()
        #     return

        # Pandas DataFrame
        elif isinstance(obj, pd.DataFrame):
            p("DataFrame Preview", color=color2)
            print(obj.head(show))  # can't replace this with p() due to formatting
            if schema:
                p("Number of columns", len(obj.columns))
                p("Number of rows", len(obj))
                p("Summary Stats")
                print(obj.describe())  # same here
                p("Schema (dtypes):", "")
                print(obj.dtypes)
            p()
            return

        # Generic value printing
        # Generic value printing
        if value is not None:
            if isinstance(value, numbers.Number):
                val = float(value)
                if float(val).is_integer():
                    formatted = f"{int(val):,}"
                else:
                    formatted = f"{val:,.{precision}f}"

                if str(obj) == "":
                    print(f"\033[{_color}m{value}\033[0m")
                else:
                    print(f"\033[{_color}m{obj}:\033[0m \033[{_color2};1m{formatted}\033[0m")

            else:
                if str(obj) == "":
                    print(f"\033[{_color}m{value}\033[0m")
                else:
                    print(f"\033[{_color};1m{obj}:\033[0m \033[{_color2}m{value}\033[0m")

        elif obj:
            print(f"\033[{_color}m\033[1m\n--- {obj} ---\033[0m")
        else:
            print()

    except Exception as e:
        print("Exception occurred", str(e))
        print("Type of object passed", type(obj))
        raise


def format_number(n):
    if n == 0:
        return "0"
    elif n < 1:
        return f"{n:.1e}"
    elif n >= 1_000_000:  # 1e6
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:  # 1e3
        return f"{n / 1_000:.2f}K"
    else:
        # return f"{n:,.2f}"
        return f"{n}"


def init_this_notebook(SEED=42):
    p("Python", sys.version)
    # p("Tensorflow", tf.__version__)
    p("Numpy", np.__version__)
    p("Panda", pd.__version__)

    # ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

    # set default plot renderer to png
    pio.renderers.default = "png"

    r.seed(SEED)
    np.random.seed(SEED)
    # tf.random.set_seed(SEED)

    p()


def normalize_paths(config: dict) -> dict:
    """Convert all paths in config['paths'] to Path objects, resolving relative to root."""

    paths = config.get("paths", {})
    root = Path(paths.get("root", ".")).resolve()

    normalized = {}
    for key, val in paths.items():
        # path = Path(val) if not isinstance(val, Path) else val
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


def normalize_type(obj: Any) -> Any:
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
        # Try int first
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
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
        return {k: normalize_type(v) for k, v in obj.items()}

    # Convert Path and bytes
    if isinstance(obj, Path):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    # Everything else stays as-is
    return obj



def format_number(n):
    if n == 0:
        return "0"
    elif n < 1:
        return f"{n:.1e}"
    elif n >= 1_000_000:  # 1e6
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:  # 1e3
        return f"{n / 1_000:.2f}K"
    else:
        # return f"{n:,.2f}"
        return f"{n}"


def init_this_notebook(SEED=42):
    p("Python", sys.version)
    # p("Tensorflow", tf.__version__)
    p("Numpy", np.__version__)
    p("Panda", pd.__version__)

    # ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

    # set default plot renderer to png
    pio.renderers.default = "png"

    r.seed(SEED)
    np.random.seed(SEED)
    # tf.random.set_seed(SEED)

    p()


def normalize_paths(config: dict) -> dict:
    """Convert all paths in config['paths'] to Path objects, resolving relative to root."""

    paths = config.get("paths", {})
    root = Path(paths.get("root", ".")).resolve()

    normalized = {}
    for key, val in paths.items():
        # path = Path(val) if not isinstance(val, Path) else val
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


def normalize_type(obj: Any) -> Any:
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
        # Try int first
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
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
        return {k: normalize_type(v) for k, v in obj.items()}

    # Convert Path and bytes
    if isinstance(obj, Path):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    # Everything else stays as-is
    return obj
