# ### In a notebook
#
# ```
# from src.utils.config import load_config
#
# config = load_config()
# print(config["paths"]["train_images"])
# ```
#
# ### In a script
# ```
# from utils.config import load_config
#
# config = load_config()
# print(config["params"]["batch_size"])
#
# ```
#

import hashlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.helpers import normalize_paths, normalize_type, p


def check_for_version(cfg, versions_dir="versions"):
    """
    Checks the current cfg against existing version snapshots.
    Creates a new version file if config changes are detected.
    Returns (version_name, version_path).
    """

    def make_json_safe(obj):
        """Convert Path and unsupported types to JSON-safe representations."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    os.makedirs(versions_dir, exist_ok=True)

    # Extract config as a clean dictionary
    cfg_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    safe_cfg = make_json_safe(cfg_dict)

    # Generate hash and JSON
    cfg_json = json.dumps(safe_cfg, sort_keys=True, indent=2)
    cfg_hash = hashlib.md5(cfg_json.encode("utf-8")).hexdigest()

    # Find existing versions
    version_files = sorted(Path(versions_dir).glob("version_*.json"))
    latest_version = None
    latest_hash = None

    if version_files:
        latest_version = version_files[-1]
        with open(latest_version, "r") as f:
            saved = json.load(f)
            latest_hash = saved.get("hash")

    # Determine whether to create or reuse version
    if not version_files:
        version_name = "v001"
    elif latest_hash != cfg_hash:
        version_num = int(latest_version.stem.split("_")[1][1:]) + 1
        version_name = f"v{version_num:03d}"
    else:
        version_name = latest_version.stem.split("_")[1]
        p(f"Config matches {version_name}", "Continuing with this version.")
        return version_name, latest_version

    # Write new version file
    version_path = Path(versions_dir) / f"version_{version_name}.json"
    with open(version_path, "w") as f:
        json.dump(
            {
                "config": safe_cfg,
                "hash": cfg_hash,
                "created": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    p("Created new version", version_name)
    return version_name, version_path


def in_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""

    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__
        # Jupyter notebook
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def get_root(marker="config.yaml"):
    """Find project root automatically."""

    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()

    if in_notebook():
        start = Path.cwd()
    else:
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Could not retrieve the current frame.")
        file_path = Path(inspect.getfile(frame)).resolve()

        start = file_path.parent

    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent.resolve()

    return start.resolve()


def load_config():
    """Load YAML configuration from project root."""

    load_dotenv()

    ROOT = get_root()
    config_path = ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert to Path objects
    config = normalize_paths(config)

    p(config)

    return ConfigNamespace(config)


class ConfigNamespace:
    def __init__(self, config: dict, prefix_sep: str = "_"):
        self._created_vars = []

        for section, entries in config.items():
            if isinstance(entries, dict):
                for key, value in entries.items():
                    if key == "root":
                        var_name = key
                    else:
                        var_name = (
                            f"{section}{prefix_sep}{key}"
                            if section != "params"
                            else key
                        )

                    var_name = var_name.upper()
                    setattr(self, var_name, normalize_type(value))
                    self._created_vars.append(var_name)

    def show(self):
        print("Injected config variables:")
        for name in self._created_vars:
            val = getattr(self, name)
            p(f"- {name}", f"{val}, {type(val)}")
        p("")


# def inject_config_vars(config: dict, prefix_sep: str = "_", target_globals: dict = None):
#     """
#     Automatically create Python variables from nested config keys.
#     """
#     # if target_globals is None:
#     #     import inspect
#     #     # Inject into the caller's global scope
#     #     target_globals = inspect.stack()[1].frame.f_globals

#     created_vars = []

#     for section, entries in config.items():
#         if isinstance(entries, dict):
#             for key, value in entries.items():
#                 if key == "root":
#                     var_name = f"{key}"
#                 else:
#                     if section != "params":
#                         var_name = f"{section}{prefix_sep}{key}"
#                     else:
#                         var_name = f"{key}"

#                 var_name = var_name.upper()
#                 #target_globals[var_name] = normalize_type(value)
#                 created_vars.append(var_name)

#     print("Injected variables:")
#     for name in created_vars:
#         p(f"- {name}", f"{target_globals[name]}, {type(target_globals[name])}")

#     p("")
#     return created_vars
