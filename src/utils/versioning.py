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
