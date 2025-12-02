
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