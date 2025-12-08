import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

from src.utils.helpers import p, t


class PathsConfig(BaseModel):
    """
    Pydantic model for managing project directory and file paths.
    
    Automatically expands relative paths to absolute paths using the root directory.
    Supports optional paths for training data, evaluation data, outputs, and models.
    All paths are validated and resolved during initialization.
    """
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
    template:Optional[Path] = None

    @validator("*", pre = True)
    def expand_paths( cls, value, values ):
        """
        Validator to expand relative paths to absolute paths.
        
        Args:
            value: The path value to validate and expand.
            values: Previously validated field values (includes 'root').
        
        Returns:
            Path: Absolute path, or None if value is None.
        """
        if value is None:
            return None
        root = values.get("root", None)
        value = Path(value)
        if not value.is_absolute() and root is not None:
            return (root / value).resolve()
        return value.resolve()

class TrainConfig(BaseModel):
    """
    Pydantic model for training hyperparameters and configuration.
    
    Contains all settings needed for model training including image size, batch size,
    learning rate, optimization schedules, and early stopping parameters.
    """
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
        """
        Load configuration from YAML file with path expansion and validation.
        
        Args:
            yaml_path (Path, optional): Path to config.yaml file. If None, uses PROJECT_ROOT environment variable.
            root (Path): Required root directory for relative path resolution.
        
        Returns:
            Config: Instantiated and validated configuration object.
        
        Raises:
            ValueError: If root is not provided.
            FileNotFoundError: If config file does not exist.
        """

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
        Print all configuration fields in a readable, formatted manner.
        
        Removes project root prefix from paths for cleaner display and shows
        training parameters, paths, and extra configuration separately.
        
        Returns:
            None (prints to console).
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
        """
        Auto-adjust configuration based on image size and available memory.
        
        Currently disabled but can implement automatic batch size and worker reductions
        for large image sizes to prevent out-of-memory errors.
        
        Returns:
            None (modifies config in-place if enabled).
        """
        # if self.train.image_size >= 512:
        #     if self.train.batch_size > 4:
        #         p("WARNING", f"Batch size {self.train.batch_size} too large for image size "
        #                      f"{self.train.image_size}. "
        #                      f"Reducing batch size to 4 to prevent OOM", color1 = c.RED)
        #         self.train.batch_size = 4
        #
        #     # Also reduce workers for large images
        #     if self.train.num_workers > 2:
        #         p("WARNING", f"Num of Workers size {self.train.num_workers} too large for image "
        #                      f"size {self.train.image_size}. "
        #                      f"Reducing num_workers to 2 to prevent OOM", color1 = c.RED)
        #         self.train.num_workers = 2

    @property
    def MASK_COLORS(self):
        """
        Get mask colors for visualization by class.
        
        Returns default colors for individual_tree and group_of_trees classes,
        merged with any user-defined colors from config.
        
        Returns:
            Dict: Mapping of class names to RGB color values [R, G, B].
        """
        default_colors = {
            "individual_tree": [0, 255, 0],
            "group_of_trees": [255, 0, 0]
        }
        user_colors = self.extra.get("MASK_COLORS", {})
        merged = default_colors.copy()
        merged.update(user_colors)
        return merged
