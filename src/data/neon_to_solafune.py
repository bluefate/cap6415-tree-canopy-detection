"""Convert NeonTreeEvaluation crown subset into Solafune-shaped RGB + JSON."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from src.utils.dataset_source import download_dataset, get_dataset_block, resolve_download_dir
from src.utils.helpers import c, p, t

CM_RESOLUTION = 10
SCENE_TYPE = "rural_area"
CLASS_NAME = "individual_tree"
# U-Net skip connections need H/W divisible by 2^4 (=16).
UNET_ALIGN = 16


def _ceil_multiple(n: int, multiple: int = UNET_ALIGN) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _pad_to_unet(img: Image.Image) -> Image.Image:
    """Pad bottom/right so H and W are multiples of 16 (polygon coords unchanged)."""
    w, h = img.size
    tw, th = _ceil_multiple(w), _ceil_multiple(h)
    if (tw, th) == (w, h):
        return img
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


def _find_neon_roots(extract_dir: Path) -> Tuple[Path, Path]:
    rgb_dirs = [p for p in extract_dir.rglob("RGB") if p.is_dir()]
    mask_dirs = [p for p in extract_dir.rglob("Masks") if p.is_dir()]
    if not rgb_dirs or not mask_dirs:
        raise FileNotFoundError(
            f"Expected RGB/ and Annotations/Masks/ under {extract_dir}"
        )
    return rgb_dirs[0], mask_dirs[0]


def _unzip(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.rglob("*.tif")):
        p("extract", "already present", color1=c.GREEN)
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    p("extracting", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    for junk in extract_dir.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)


def _instance_to_polygon(mask: np.ndarray) -> List[float]:
    """Binary (H, W) mask → flat [x, y, ...] convex hull polygon."""
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < 3:
        return []
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    if len(pts) > 2000:
        idx = np.linspace(0, len(pts) - 1, 2000).astype(int)
        pts = pts[idx]
    if len(np.unique(pts, axis=0)) < 3:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        return [x0, y0, x1, y0, x1, y1, x0, y1]
    try:
        hull = ConvexHull(pts)
    except Exception:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        return [x0, y0, x1, y0, x1, y1, x0, y1]
    poly: List[float] = []
    for i in hull.vertices:
        poly.extend([float(pts[i, 0]), float(pts[i, 1])])
    return poly if len(poly) >= 6 else []


def _mask_to_polygons(instance_masks: np.ndarray) -> List[List[float]]:
    """instance_masks: (N, H, W) bool → list of flat [x,y,...] polygons."""
    if instance_masks.ndim == 2:
        instance_masks = instance_masks[None, ...]
    polygons: List[List[float]] = []
    for i in range(instance_masks.shape[0]):
        poly = _instance_to_polygon(instance_masks[i])
        if poly:
            polygons.append(poly)
    return polygons


def _stem_pairs(rgb_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs = []
    for rgb in sorted(rgb_dir.glob("*.tif")):
        mask = mask_dir / f"{rgb.stem}.npy"
        if mask.exists():
            pairs.append((rgb.stem, rgb, mask))
        else:
            p("skip (no mask)", rgb.name, color1=c.ORANGE)
    return pairs


def _write_png(rgb_path: Path, dest: Path) -> Tuple[int, int]:
    img = _pad_to_unet(Image.open(rgb_path).convert("RGB"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    w, h = img.size
    return w, h


def _image_record(
    file_name: str,
    width: int,
    height: int,
    polygons: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "width": width,
        "height": height,
        "cm_resolution": CM_RESOLUTION,
        "scene_type": SCENE_TYPE,
        "annotations": [
            {
                "class": CLASS_NAME,
                "confidence_score": 1.0,
                "segmentation": list(poly),
            }
            for poly in polygons
        ],
    }


def build_public_sample(
    config,
    *,
    force: bool = False,
    eval_count: int = 4,
) -> Dict[str, Any]:
    """
    Download (if needed), convert NEON subset → data/public_sample Solafune layout.

    Uses config paths after public source merge (train_images, eval_images, annotations).
    """
    t("Build public NEON sample")
    # Never force-redownload here — force only rebuilds PNGs/JSON from the local zip.
    download_dataset(config, force=False, source_key="public")

    sources = get_dataset_block(config).get("sources") or {}
    public = {"key": "public", **(sources.get("public") or {})}
    raw_dir = resolve_download_dir(config, public)
    zip_path = raw_dir / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}; download failed")

    extract_dir = raw_dir / "extracted"
    _unzip(zip_path, extract_dir)
    rgb_dir, mask_dir = _find_neon_roots(extract_dir)
    pairs = _stem_pairs(rgb_dir, mask_dir)
    if not pairs:
        raise RuntimeError("No RGB/mask pairs found in NEON subset")

    root = Path(config.paths.root)
    train_dir = Path(config.paths.train_images)
    eval_dir = Path(config.paths.eval_images)
    ann_path = Path(config.paths.annotations)
    template_path = Path(config.paths.template)

    if "public_sample" not in str(train_dir):
        train_dir = root / "data/public_sample/train_images"
        eval_dir = root / "data/public_sample/evaluation_images"
        ann_path = root / "data/public_sample/train_annotations.json"
        template_path = root / "data/public_sample/sample_answer.json"

    if force:
        for d in (train_dir, eval_dir):
            if d.exists():
                shutil.rmtree(d)

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    n_eval = min(eval_count, max(1, len(pairs) // 4))
    eval_stems = {stem for stem, _, _ in pairs[-n_eval:]}
    train_images: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []

    for stem, rgb_path, mask_path in pairs:
        masks = np.load(mask_path)
        polygons = _mask_to_polygons(masks)
        out_name = f"neon_{stem}.png"
        is_eval = stem in eval_stems
        dest = (eval_dir if is_eval else train_dir) / out_name
        width, height = _write_png(rgb_path, dest)
        record = _image_record(out_name, width, height, polygons)
        if is_eval:
            eval_records.append(record)
        else:
            train_images.append(record)
        p(
            "converted",
            f"{out_name} ({'eval' if is_eval else 'train'}) "
            f"trees={len(polygons)} {width}x{height}",
        )

    sample_answer = {"images": [{**rec, "annotations": []} for rec in eval_records]}

    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps({"images": train_images}))
    template_path.write_text(json.dumps(sample_answer))
    ann_path.with_name("evaluation_annotations.json").write_text(
        json.dumps({"images": eval_records})
    )

    summary = {
        "train_images": len(train_images),
        "eval_images": len(eval_records),
        "train_trees": sum(len(im["annotations"]) for im in train_images),
        "eval_trees": sum(len(im["annotations"]) for im in eval_records),
        "annotations": str(ann_path),
        "template": str(template_path),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
    }
    p(
        "Done",
        f"train={summary['train_images']} eval={summary['eval_images']} "
        f"trees={summary['train_trees']}+{summary['eval_trees']}",
        color1=c.GREEN,
    )
    return summary
