from pathlib import Path

import cv2
import numpy as np

from src.data.annotations import AnnotationEntry
from src.data.image_loader import load_image as robust_load_image
from src.exploration.visualize import (
    show_side_by_side,
)
from src.utils.config import Config
from src.utils.helpers import p


# -----------------------------------------------------------
# Core helpers
# -----------------------------------------------------------
config = Config.load()
CLASS_NAMES = ["individual_tree", "group_of_trees"]


# def load_image(image_dir: Path, entry: AnnotationEntry):
#     path = image_dir / entry.image_path.name
#     img = cv2.imread(str(path))
#     if img is None:
#         raise RuntimeError("Failed to read image " + str(path))
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def load_image(image_dir: Path, entry: AnnotationEntry):
    path = image_dir / entry.image_path.name
    return robust_load_image(path)


def mask_for_class(entry: AnnotationEntry, cls: str):
    H = entry.height
    W = entry.width
    mask = np.zeros((H, W), dtype=np.uint8)

    for item in entry.items:
        if item.cls != cls:
            continue
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)

    return mask


def available_classes(entries):
    classes = set()
    for e in entries:
        for item in e.items:
            classes.add(item.cls)
    return sorted(classes)


def mask_all(entry: AnnotationEntry):
    H = entry.height
    W = entry.width
    mask = np.zeros((H, W), dtype=np.uint8)
    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    return mask


# -----------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------


def color_mask(entry: AnnotationEntry):
    H = entry.height
    W = entry.width
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        cls = item.cls
        color = config.MASK_COLORS.get(cls, (255, 255, 255))
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask_rgb, [poly], color)

    return mask_rgb


def draw_bboxes(img, entry: AnnotationEntry):
    out = img.copy()
    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        xs = seg[0::2]
        ys = seg[1::2]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        if item.cls == "individual_tree":
            color = (0, 255, 0)
        elif item.cls == "group_of_trees":
            color = (255, 0, 0)
        else:
            color = (0, 128, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = item.cls
        cv2.putText(
            out, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )
    return out


def draw_bboxes_for_class(img, entry: AnnotationEntry, cls: str):
    out = img.copy()
    for item in entry.items:
        if item.cls != cls:
            continue
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        xs = seg[0::2]
        ys = seg[1::2]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        if cls == "individual_tree":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            out, cls, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 3
        )

    return out


def explore_image(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    mask_ind = mask_for_class(entry, "individual_tree")
    mask_grp = mask_for_class(entry, "group_of_trees")

    color_ind = np.zeros_like(img)
    color_grp = np.zeros_like(img)

    col_ind = tuple(config.MASK_COLORS.get("individual_tree", [0, 255, 0]))
    col_grp = tuple(config.MASK_COLORS.get("group_of_trees", [255, 0, 0]))

    color_ind[mask_ind == 1] = col_ind
    color_grp[mask_grp == 1] = col_grp

    # over_ind = show_overlay(img, mask_ind, return_img=True)
    # over_grp = show_overlay(img, mask_grp, return_img=True)
    over_ind = cv2.addWeighted(img, 0.6, color_ind, 0.4, 0)
    over_grp = cv2.addWeighted(img, 0.6, color_grp, 0.4, 0)

    show_side_by_side(
        # img,
        (mask_ind, "individual_tree"),
        (mask_grp, "group_of_trees"),
        over_ind,
        over_grp,
        titles=(
            # "Image",
            "Mask individual",
            "Mask group",
            "Overlay individual",
            "Overlay group",
        ),
        mask_colors=config.MASK_COLORS,
    )


def show_single_class(entry: AnnotationEntry, image_dir: Path, cls: str):
    img = load_image(image_dir, entry)
    mask = mask_for_class(entry, cls)
    show_side_by_side(
        img,
        (mask, cls),
        titles=("Image", f"Mask {cls}"),
        mask_colors=config.MASK_COLORS,
    )


def show_all_classes(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    # build combined mask for both classes
    mask = mask_all(entry)

    # build colored mask (per polygon, per class)
    mask_rgb = color_mask(entry)

    # overlay with mask colors
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img,
        mask_rgb,
        overlay,
        titles=("Image", "Mask All (color)", "Overlay All"),
        mask_colors=config.MASK_COLORS,
    )


def show_per_class(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    classes = ["individual_tree", "group_of_trees"]

    # masks must be tuples (mask, class_name) for coloring
    imgs = [img] + [(mask_for_class(entry, c), c) for c in classes]

    titles = ["Image"] + classes

    show_side_by_side(
        *imgs,
        titles=tuple(titles),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_all(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    # mask = mask_all(entry)
    mask = color_mask(entry)

    # show_overlay(img, mask, title="Overlay All")
    overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

    show_side_by_side(
        overlay,
        titles=("Overlay All",),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_by_class(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    m_ind = mask_for_class(entry, "individual_tree")
    m_grp = mask_for_class(entry, "group_of_trees")

    col_ind = tuple(config.MASK_COLORS.get("individual_tree", [0, 255, 0]))
    col_grp = tuple(config.MASK_COLORS.get("group_of_trees", [255, 0, 0]))

    # color masks
    color_ind = np.zeros_like(img)
    color_grp = np.zeros_like(img)

    color_ind[m_ind == 1] = col_ind
    color_grp[m_grp == 1] = col_grp

    # overlays
    overlay_ind = cv2.addWeighted(img, 0.6, color_ind, 0.4, 0)
    overlay_grp = cv2.addWeighted(img, 0.6, color_grp, 0.4, 0)

    show_side_by_side(
        overlay_ind,
        overlay_grp,
        titles=("Overlay individual", "Overlay group"),
        mask_colors=config.MASK_COLORS,
    )


# -----------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------


def explore_color_overlay(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    mask_rgb = color_mask(entry)
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img, mask_rgb, overlay, titles=("Image", "Color mask", "Overlay color mask")
    )


def explore_bboxes(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    b_all = draw_bboxes(img, entry)
    b_ind = draw_bboxes_for_class(img, entry, "individual_tree")
    b_grp = draw_bboxes_for_class(img, entry, "group_of_trees")

    show_side_by_side(
        b_ind, b_grp, b_all, titles=("Individual bboxes", "Group bboxes", "All bboxes")
    )


def find_images_with_both(entries):
    out = []
    for e in entries:
        classes = {item.cls for item in e.items}
        if "individual_tree" in classes and "group_of_trees" in classes:
            out.append(e)
    return out


def count_classes(entry: AnnotationEntry):
    counts = {"individual_tree": 0, "group_of_trees": 0}
    for item in entry.items:
        if item.cls in counts:
            counts[item.cls] += 1
    return counts


def class_distribution(entries):
    if isinstance(entries, AnnotationEntry):
        entries = [entries]
    total = {"individual_tree": 0, "group_of_trees": 0}
    for e in entries:
        for item in e.items:
            if item.cls in total:
                total[item.cls] += 1
    return total


def dataset_report(entries, image_dir: Path, sample_count=3):
    dist = class_distribution(entries)
    p("Class counts:", dist)

    both = find_images_with_both(entries)
    p("Images containing both classes:", len(both))

    samples = entries[:sample_count]

    for e in samples:
        p("Image:", e.image_path.name)
        explore_color_overlay(e, image_dir)
        explore_bboxes(e, image_dir)
        explore_image(e, image_dir)
