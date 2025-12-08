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
        Initialize version manager for a directory.
        
        Args:
            base_dir (Path): Parent directory that holds version folders (e.g., checkpoints or runs).
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents = True, exist_ok = True)


    def _normalize_for_json( self, obj ):
        """
        Recursively convert objects to JSON-serializable types.
        
        Converts Path objects to strings and handles nested structures.
        
        Args:
            obj: Object to normalize.
        
        Returns:
            JSON-serializable version of the object.
        """
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
        Compute a stable MD5 hash of the configuration dictionary.
        
        Normalizes config to JSON and sorts keys for consistent hashing.
        
        Args:
            cfg (Dict[str, Any]): Configuration dictionary to hash.
        
        Returns:
            str: Hexadecimal MD5 hash of the configuration.
        """
        clean = self._normalize_for_json(cfg)
        cfg_json = json.dumps(clean, sort_keys = True)
        return hashlib.md5(cfg_json.encode("utf8")).hexdigest()

    def find_latest( self ) -> Optional[Path]:
        """
        Return the path of the latest version folder or None if no versions exist.
        
        Versions are sorted numerically based on v001, v002, etc. naming.
        
        Returns:
            Optional[Path]: Path to latest version folder, or None if none exist.
        """
        versions = sorted(self.base_dir.glob("v*"))
        if not versions:
            return None
        return versions[-1]

    def create_next_version( self ) -> Path:
        """
        Create and return the next version folder.
        
        Increments the latest version number (v001 -> v002) or creates v001 if none exist.
        
        Returns:
            Path: Path to newly created version folder.
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
        Determine the correct version folder for a configuration.
        
        If the latest version has a matching config hash, reuses it.
        Otherwise, creates a new version with the config snapshot.
        
        Args:
            cfg (Dict[str, Any]): Configuration dictionary to resolve.
        
        Returns:
            Path: Version folder path (existing or newly created).
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
        Save configuration and hash to version directory for reproducibility.
        
        Creates a config_snapshot.json file with config, hash, and creation timestamp.
        
        Args:
            version_dir (Path): Directory to save snapshot to.
            cfg (Dict[str, Any]): Configuration dictionary to save.
            cfg_hash (str): Configuration hash.
        
        Returns:
            None
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
        Produce standard file paths used by training and evaluation workflows.
        
        Returns a dictionary with paths for checkpoints, best model, metrics, and logs.
        
        Args:
            version_dir (Path): Version directory.
        
        Returns:
            Dict[str, Path]: Mapping of purpose to file paths.
        """
        return {
            "checkpoint": version_dir / "checkpoint.pth",
            "best": version_dir / "best_model.pth",
            "metrics": version_dir / "metrics.csv",
            "log": version_dir / "run.log",
        }
