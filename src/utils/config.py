import hashlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.helpers import make_json_safe, normalize_paths, normalize_type, p


def check_for_version( cfg, versions_dir = "versions" ):
    """
    Checks the current cfg against existing version snapshots.
    Creates a new version file if config changes are detected.
    Returns (version_name, version_path).
    """

    os.makedirs(versions_dir, exist_ok = True)

    # Extract config as a clean dictionary
    cfg_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    safe_cfg = make_json_safe(cfg_dict)

    # Generate hash and JSON
    cfg_json = json.dumps(safe_cfg, sort_keys = True, indent = 2)
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
            indent = 2,
        )

    p("Created new version", version_name)
    return version_name, version_path


def in_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


class Config:
    """Singleton configuration loader."""
    _instance = None
    _created_vars = []

    def __new__( cls ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    @staticmethod
    def _get_root( marker = "config.yaml" ):
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

    def _load_config( self ):
        """Load YAML configuration from project root."""
        load_dotenv()
        ROOT = self._get_root()
        config_path = ROOT / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Convert to Path objects
        config = normalize_paths(config)

        # Process config into attributes
        self._process_config(config)

    def _process_config( self, config: dict, prefix_sep: str = "_" ):
        """Process config dictionary into class attributes."""
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

    def show( self ):
        """Display all config variables."""
        p("Injected config variables:")
        for name in self._created_vars:
            val = getattr(self, name)
            p(f"- {name}", f"{val}, {type(val)}")
        p("")

    def __str__( self ):
        """Return string representation of config."""
        lines = ["Config:"]
        for name in self._created_vars:
            val = getattr(self, name)
            lines.append(f"  {name}: {val}")
        return "\n".join(lines)

    def __repr__( self ):
        """Return detailed representation."""
        return f"Config(loaded with {len(self._created_vars)} attributes)"
