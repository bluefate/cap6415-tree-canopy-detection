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

# +
from dotenv import load_dotenv
from pathlib import Path
from src.utils.helpers import normalize_paths, normalize_type, p

import inspect
import os
import yaml


def in_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""

    try:
        from IPython import get_ipython

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

    return config


def inject_config_vars(config: dict, prefix_sep: str = "_", target_globals: dict = None):
    """
    Automatically create Python variables from nested config keys.
    """
    if target_globals is None:
        import inspect

        # Inject into the caller's global scope
        target_globals = inspect.stack()[1].frame.f_globals

    created_vars = []

    for section, entries in config.items():
        if isinstance(entries, dict):
            for key, value in entries.items():
                if key == "root":
                    var_name = f"{key}"
                else:
                    if section != "params":
                        var_name = f"{section}{prefix_sep}{key}"
                    else:
                        var_name = f"{key}"

                var_name = var_name.upper()
                target_globals[var_name] = normalize_type(value)
                created_vars.append(var_name)

    print("Injected variables:")
    for name in created_vars:
        p(f"- {name}", f"{target_globals[name]}, {type(target_globals[name])}")

    p("")
    # return created_vars
