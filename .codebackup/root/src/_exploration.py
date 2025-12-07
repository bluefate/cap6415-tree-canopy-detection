# From C:\github\Tree-Canopy-Detection\src\exploration\class_explorer.py
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data.annotations import AnnotationEntry
from src.data.image_loader import load_image
from src.exploration.visualize import (
    show_side_by_side,
)
from src.utils.helpers import p, t


# -----------------------------------------------------------
# Core helpers
# -----------------------------------------------------------
CLASS_NAMES = ["individual_tree", "group_of_trees"]


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


def color_mask(config, entry: AnnotationEntry):
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


def explore_image(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)

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


def analyze_class_distribution(train_loader, val_loader=None):
    """
    Analyze class distribution in training (and optionally validation) data.
    Call this before training to understand class imbalance and get weight recommendations.
    """

    p()
    t("CLASS DISTRIBUTION ANALYSIS")

    def analyze_loader(loader, name):
        all_masks = []
        total_images = 0

        for _, mask in loader:
            all_masks.append(mask.flatten())
            total_images += mask.shape[0]

        all_masks = torch.cat(all_masks)
        counts = torch.bincount(all_masks, minlength=3)
        total_pixels = counts.sum().item()

        print(f"\n{name}:")
        print(f"  Total images: {total_images}")
        print(f"  Total pixels: {total_pixels:,}")
        print()

        class_names = ["background", "individual_tree", "group_of_trees"]
        percentages = []

        for i, (class_name, count) in enumerate(zip(class_names, counts)):
            pct = count.item() / total_pixels * 100
            percentages.append(pct)
            print(
                f"  Class {i} ({class_name:15}): {count.item():>10,} pixels ({pct:5.2f}%)"
            )

        return counts, percentages

    # Analyze training data
    train_counts, train_pcts = analyze_loader(train_loader, "TRAINING SET")

    # Analyze validation data if provided
    if val_loader is not None:
        val_counts, val_pcts = analyze_loader(val_loader, "VALIDATION SET")

    # Calculate recommended weights (inverse frequency)
    p()
    t("RECOMMENDED WEIGHTS")

    total = train_counts.sum().float()
    frequencies = train_counts.float() / total

    # Method 1: Inverse frequency
    inv_freq_weights = 1.0 / (frequencies + 1e-8)
    inv_freq_weights = (
        inv_freq_weights / inv_freq_weights.sum() * 3
    )  # Normalize to sum=3

    # Method 2: Balanced weights (sklearn-style)
    n_classes = 3
    n_samples = total.item()
    balanced_weights = n_samples / (n_classes * train_counts.float() + 1e-8)
    balanced_weights = balanced_weights / balanced_weights.min()  # Normalize so min=1

    # Method 3: Simple practical weights (background down, trees up)
    bg_ratio = train_pcts[0] / 100
    simple_weights = torch.tensor(
        [
            0.3,  # Background (reduce)
            1.0 / (train_pcts[1] / 100 + 0.1),  # Individual tree
            1.0 / (train_pcts[2] / 100 + 0.1),  # Group of trees
        ]
    )
    simple_weights = simple_weights / simple_weights.sum() * 3

    # using print because p nt outputing weights correctly.
    print("\nMethod 1 - Inverse Frequency:")
    print(
        f"  weights = torch.tensor([{inv_freq_weights[0]:.2f}, {inv_freq_weights[1]:.2f}, {inv_freq_weights[2]:.2f}])"
    )

    print("\nMethod 2 - Balanced (sklearn-style):")
    print(
        f"  weights = torch.tensor([{balanced_weights[0]:.2f}, {balanced_weights[1]:.2f}, {balanced_weights[2]:.2f}])"
    )

    print("\nMethod 3 - Simple Practical:")
    print(
        f"  weights = torch.tensor([{simple_weights[0]:.2f}, {simple_weights[1]:.2f}, {simple_weights[2]:.2f}])"
    )

    return {
        "train_counts": train_counts,
        "train_percentages": train_pcts,
        "inv_freq_weights": inv_freq_weights,
        "balanced_weights": balanced_weights,
        "simple_weights": simple_weights,
    }


def show_single_class(config, entry: AnnotationEntry, cls: str):
    img = load_image(config.paths.train_images, entry)
    mask = mask_for_class(entry, cls)
    show_side_by_side(
        img,
        (mask, cls),
        titles=("Image", f"Mask {cls}"),
        mask_colors=config.MASK_COLORS,
    )


def show_all_classes(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)

    # build combined mask for both classes
    mask = mask_all(entry)

    # build colored mask (per polygon, per class)
    mask_rgb = color_mask(config, entry)

    # overlay with mask colors
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img,
        mask_rgb,
        overlay,
        titles=("Image", "Mask All (color)", "Overlay All"),
        mask_colors=config.MASK_COLORS,
    )


def show_per_class(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)
    classes = ["individual_tree", "group_of_trees"]

    # masks must be tuples (mask, class_name) for coloring
    imgs = [img] + [(mask_for_class(entry, c), c) for c in classes]

    titles = ["Image"] + classes

    show_side_by_side(
        *imgs,
        titles=tuple(titles),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_all(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)
    # mask = mask_all(entry)
    mask = color_mask(config, entry)

    # show_overlay(img, mask, title="Overlay All")
    overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

    show_side_by_side(
        overlay,
        titles=("Overlay All",),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_by_class(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)

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


def explore_color_overlay(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)
    mask_rgb = color_mask(config, entry)
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img, mask_rgb, overlay, titles=("Image", "Color mask", "Overlay color mask")
    )


def explore_bboxes(config, entry: AnnotationEntry):
    img = load_image(config.paths.train_images, entry)
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


def dataset_report(config, entries, sample_count=3):
    dist = class_distribution(entries)
    p("Class counts", dist)

    both = find_images_with_both(entries)
    p("Images containing both classes:", len(both))

    samples = entries[:sample_count]

    for e in samples:
        p()
        t(f"Image: {e.image_path.name}")
        explore_color_overlay(config, e)
        explore_bboxes(config, e)
        explore_image(config, e)
        p()


def create_experiment_tracker():
    """Create a tracker to store and plot experiment results."""
    return {
        "models": [],
        "val_loss": [],
        "iou": [],
        "iou_individual": [],
        "iou_group": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }


def update_tracker(tracker, model_name, checkpoint_path):
    """Update tracker with results from a completed experiment."""
    if not checkpoint_path.exists():
        return False

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        tracker["models"].append(model_name)
        tracker["val_loss"].append(ckpt.get("val_loss", ckpt.get("best_val_loss", 0)))
        tracker["iou"].append(ckpt.get("val_iou", ckpt.get("iou", 0)))
        tracker["iou_individual"].append(ckpt.get("val_iou_individual", 0))
        tracker["iou_group"].append(ckpt.get("val_iou_group", 0))
        tracker["precision"].append(ckpt.get("val_precision", ckpt.get("precision", 0)))
        tracker["recall"].append(ckpt.get("val_recall", ckpt.get("recall", 0)))
        tracker["f1"].append(ckpt.get("val_f1_score", ckpt.get("f1_score", 0)))
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def plot_experiment_results(tracker, save_path=None):
    """Plot comparison of all completed experiments."""
    if len(tracker["models"]) == 0:
        print("No experiments completed yet.")
        return

    # One row, six columns
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")

    models = tracker["models"]
    x = range(len(models))
    colors = plt.cm.tab10(range(len(models)))

    def setup_xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center", fontsize=8)

    # Plot 1: Val Loss (lower is better)
    ax = axes[0]
    ax.bar(x, tracker["val_loss"], color=colors)
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss (lower=better)")
    setup_xticks(ax)
    ax.axhline(y=min(tracker["val_loss"]), color="green", linestyle="--", alpha=0.5)

    # Plot 2: IoU (higher is better)
    ax = axes[1]
    ax.bar(x, tracker["iou"], color=colors)
    ax.set_ylabel("IoU")
    ax.set_title("Mean IoU (higher=better)")
    setup_xticks(ax)
    ax.axhline(y=max(tracker["iou"]), color="green", linestyle="--", alpha=0.5)

    # Plot 3: IoU by class
    ax = axes[2]
    width = 0.35
    x_arr = np.arange(len(models))
    ax.bar(
        x_arr - width / 2,
        tracker["iou_individual"],
        width,
        label="Individual",
        color="forestgreen",
    )
    ax.bar(x_arr + width / 2, tracker["iou_group"], width, label="Group", color="gold")
    ax.set_ylabel("IoU")
    ax.set_title("IoU by Class")
    ax.set_xticks(x_arr)
    ax.set_xticklabels(models, rotation=45, ha="center", fontsize=8)
    ax.legend()

    # Plot 4: Precision
    ax = axes[3]
    ax.bar(x, tracker["precision"], color=colors)
    ax.set_ylabel("Precision")
    ax.set_title("Precision (higher=better)")
    setup_xticks(ax)

    # Plot 5: Recall
    ax = axes[4]
    ax.bar(x, tracker["recall"], color=colors)
    ax.set_ylabel("Recall")
    ax.set_title("Recall (higher=better)")
    setup_xticks(ax)

    # Plot 6: F1 Score
    ax = axes[5]
    ax.bar(x, tracker["f1"], color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score (higher=better)")
    setup_xticks(ax)
    ax.axhline(y=max(tracker["f1"]), color="green", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    # Print summary table
    p()
    t(
        f"{'Model':<20} {'Loss':>8} {'IoU':>8} {'Ind':>8} {'Grp':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    )

    best_iou_idx = np.argmax(tracker["iou"])
    for i, model in enumerate(models):
        marker = " üèÜ" if i == best_iou_idx else ""
        print(
            f"{model:<20} {tracker['val_loss'][i]:>8.4f} {tracker['iou'][i]:>8.4f} "
            f"{tracker['iou_individual'][i]:>8.4f} {tracker['iou_group'][i]:>8.4f} "
            f"{tracker['precision'][i]:>8.4f} {tracker['recall'][i]:>8.4f} {tracker['f1'][i]:>8.4f}{marker}"
        )


def plot_training_history(trainer, title_prefix=""):
    history = trainer.history

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_iou = history.get("val_iou", [])
    val_precision = history.get("val_precision", [])
    val_recall = history.get("val_recall", [])
    val_f1 = history.get("val_f1", [])
    lr = history.get("lr", [])

    n_epochs = len(train_loss)
    if n_epochs == 0:
        print("No history to plot")
        return

    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"{title_prefix} epoch metrics", fontsize=14, fontweight="bold")

    # 1. Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per epoch")
    ax.legend()

    # 2. IoU and F1
    ax = axes[1]
    if len(val_iou) == n_epochs:
        ax.plot(epochs, val_iou, label="Val IoU")
    if len(val_f1) == n_epochs:
        ax.plot(epochs, val_f1, label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("IoU and F1 per epoch")
    ax.legend()

    # 3. Precision and Recall
    ax = axes[2]
    if len(val_precision) == n_epochs:
        ax.plot(epochs, val_precision, label="Val precision")
    if len(val_recall) == n_epochs:
        ax.plot(epochs, val_recall, label="Val recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall per epoch")
    ax.legend()

    # 4. Learning rate
    ax = axes[3]
    if len(lr) == n_epochs:
        ax.plot(epochs, lr, label="LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


# From C:\github\Tree-Canopy-Detection\src\exploration\content.py
import cv2
import numpy as np


def crop_bbox_with_context(image_dir, entry, item, context_factor=1.5):
    """
    Crop bbox with natural background context instead of artificial padding.

    Args:
        entry: annotation entry
        item: bbox item
        context_factor: how much extra context to include (1.5 = 50% more on each side)
    """
    img_path = image_dir / entry.image_path.name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # Get original bbox
    x1, y1, x2, y2 = item.bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate context expansion
    context_w = int(bbox_w * (context_factor - 1) / 2)
    context_h = int(bbox_h * (context_factor - 1) / 2)

    # Expand bbox with context, but clip to image boundaries
    new_x1 = max(0, x1 - context_w)
    new_y1 = max(0, y1 - context_h)
    new_x2 = min(img_w, x2 + context_w)
    new_y2 = min(img_h, y2 + context_h)

    # Crop with natural context
    crop = img[new_y1:new_y2, new_x1:new_x2]

    return crop, (new_x1, new_y1, new_x2, new_y2)


def crop_mask_with_context(entry, item, context_bbox):
    """
    Create mask for the context-expanded crop.
    """
    new_x1, new_y1, new_x2, new_y2 = context_bbox

    # Create full image mask
    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)

    # Crop mask to match context crop
    mask_crop = mask[new_y1:new_y2, new_x1:new_x2]
    return mask_crop


def resize_to_standard(img, mask, target_size=256):
    """
    Resize crop and mask to standard size for training.
    Much better than padding with zeros.
    """
    img_resized = cv2.resize(
        img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    return img_resized, mask_resized


def crop_bbox(image_dir, entry, item):
    img_path = image_dir / entry.image_path.name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = item.bbox
    return img[y1:y2, x1:x2]


def crop_mask(entry, item):
    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    x1, y1, x2, y2 = item.bbox
    return mask[y1:y2, x1:x2]


def pad_to_size(img, target_h, target_w):
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def expand_crop_with_background(img, mask, bbox, target_h, target_w):
    """
    Instead of padding with zeros, expand the crop to include more background.
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    crop_h, crop_w = y2 - y1, x2 - x1

    # Calculate how much more we need
    need_more_h = target_h - crop_h
    need_more_w = target_w - crop_w

    if need_more_h <= 0 and need_more_w <= 0:
        # Already big enough, just return the crop
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    # Expand the bbox to get more background
    expand_h = max(0, need_more_h // 2)
    expand_w = max(0, need_more_w // 2)

    # Calculate new bounds, staying within image
    new_y1 = max(0, y1 - expand_h)
    new_x1 = max(0, x1 - expand_w)
    new_y2 = min(img_h, new_y1 + target_h)
    new_x2 = min(img_w, new_x1 + target_w)

    # If we still don't have enough space, adjust the other side
    if new_y2 - new_y1 < target_h:
        new_y1 = max(0, new_y2 - target_h)
    if new_x2 - new_x1 < target_w:
        new_x1 = max(0, new_x2 - target_w)

    # Extract the expanded crop
    expanded_img = img[new_y1:new_y2, new_x1:new_x2]
    expanded_mask = mask[new_y1:new_y2, new_x1:new_x2]

    # If we still need to reach exact target size, resize
    if expanded_img.shape[:2] != (target_h, target_w):
        expanded_img = cv2.resize(
            expanded_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        expanded_mask = cv2.resize(
            expanded_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    return expanded_img, expanded_mask


# From C:\github\Tree-Canopy-Detection\src\exploration\enhancement.py
import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def equalize_hist(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image.
    """
    return cv2.equalizeHist(gray)


def clahe_enhance(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Apply a basic sharpening filter to enhance edges.
    """
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    out = cv2.filter2D(gray, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(gray: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to zero to one.
    """
    g = gray.astype(np.float32)
    m = g.min()
    M = g.max()
    if M <= m:
        return g
    return (g - m) / (M - m)


def enhance_image_for_segmentation(image: np.ndarray) -> tuple:
    """
    Full enhancement pipeline used in notebooks.
    Returns enhanced image and intermediate stages.

    Steps:
    1. convert to grayscale
    2. equalize histogram
    3. apply CLAHE
    4. sharpen
    5. normalize to zero to one
    """

    stages = {}

    gray = to_gray(image)
    stages["gray"] = gray

    eq = equalize_hist(gray)
    stages["equalized"] = eq

    clahe = clahe_enhance(eq)
    stages["clahe"] = clahe

    sharp = sharpen(clahe)
    stages["sharpened"] = sharp

    norm = normalize(sharp)
    stages["normalized"] = norm

    return norm, stages


# From C:\github\Tree-Canopy-Detection\src\exploration\evaluation.py
import torch

from src.utils.helpers import p


def load_best_model(model_name: str, config, notebook="10", mode="rgb", filters=None):
    """Load the best trained model with correct path structure."""
    from src.models.zoo import build_model
    from src.utils.versioning import VersionManager

    # Construct the exact path used by training
    model_dir = config.paths.models / notebook / model_name / mode

    # Add filters to path if used
    if filters:
        if isinstance(filters, list):
            filter_str = "_".join(filters)
        else:
            filter_str = str(filters)
        model_dir = model_dir / filter_str

    # Add image size to path
    model_dir = model_dir / f"size_{config.train.image_size}"

    p(f"Looking for models in: {model_dir}")

    if not model_dir.exists():
        print(f"Model directory doesn't exist: {model_dir}")
        # Show what actually exists
        base_dir = config.paths.models
        if base_dir.exists():
            print("Available structure:")
            for item in base_dir.rglob("*"):
                if item.is_dir() or item.suffix == ".pth":
                    print(f"  {item.relative_to(base_dir)}")
        raise RuntimeError(f"No model directory found: {model_dir}")

    # Use VersionManager to find latest version
    vm = VersionManager(model_dir)
    version_dir = vm.find_latest()

    if version_dir is None:
        print(f"No version directories found in: {model_dir}")
        print("Contents:")
        for item in model_dir.glob("*"):
            print(f"  {item.name}")
        raise RuntimeError("No trained models found")

    # Try best_model.pth first, then checkpoint.pth
    best_path = version_dir / "best_model.pth"
    checkpoint_path = version_dir / "checkpoint.pth"

    if best_path.exists():
        model_path = best_path
        print(f"Loading best model from: {best_path}")
    elif checkpoint_path.exists():
        model_path = checkpoint_path
        print(f"Loading checkpoint from: {checkpoint_path}")
    else:
        print(f"No model files found in: {version_dir}")
        print("Available files:")
        for item in version_dir.glob("*"):
            print(f"  {item.name}")
        raise RuntimeError(f"No model weights found in: {version_dir}")

    # Determine input channels based on mode and filters
    if mode == "concat" and filters:
        in_channels = 6  # RGB + 3 filters
    else:
        in_channels = 3  # RGB or filtered RGB

    # Build and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, in_channels=in_channels, out_channels=3)

    # Load weights
    state = torch.load(model_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    print(f"Successfully loaded model from: {model_path}")
    return model


def diagnose_model_directory(config):
    """Diagnose the actual directory structure."""
    models_dir = config.paths.models
    print(f"Models directory: {models_dir}")
    print(f"Exists: {models_dir.exists()}")

    if models_dir.exists():
        print("\nDirectory structure:")
        for item in models_dir.rglob("*"):
            if item.is_dir() or item.suffix == ".pth":
                depth = len(item.relative_to(models_dir).parts)
                indent = "  " * (depth - 1)
                print(f"{indent}{item.name}")


# From C:\github\Tree-Canopy-Detection\src\exploration\filters.py
import cv2
import numpy as np


def cv2_apply_gaussian(
    image: np.ndarray, ksize: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur to an RGB image.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def cv2_apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to a grayscale or RGB image.
    If RGB, converts to grayscale internally.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(dx * dx + dy * dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


def cv2_apply_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    Converts to grayscale if needed.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return out


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize to range zero to one.
    """
    img = image.astype(np.float32)
    m = img.min()
    M = img.max()
    if M <= m:
        return img
    return (img - m) / (M - m)


# From C:\github\Tree-Canopy-Detection\src\exploration\filter_utils.py
"""
Filter utilities with automatic shape normalization.

This module provides helpers for applying filters and creating multi-channel
enhanced images without shape mismatch errors.
"""

import cv2
import numpy as np

from src.utils.helpers import c, p


def get_input_channels(mode, filters, model_name):
    """Get correct input channels for model and mode combination."""
    if mode == "rgb":
        return 3
    elif mode == "filtered":
        return 3  # Filters replace RGB channels
    elif mode == "concat":
        if model_name in ["simple_cnn", "unet"]:
            # These models don't support 6 channels - use RGB instead
            p(
                "WARNING",
                f"{model_name} doesn't support concat mode, using RGB",
                color1=c.ORANGE,
            )
            return 3
        else:
            num_filters = len(filters) if filters else 0
            return 3 + num_filters  # RGB + actual filter count
    else:
        return 3  # Default fallback


def normalize_filter_output(filtered, target_shape, dtype=np.uint8):
    """
    Normalize filter output to consistent shape and dtype.

    Args:
        filtered: Filter output (any shape, any dtype)
        target_shape: Target (height, width) tuple
        dtype: Output dtype (default: np.uint8)

    Returns:
        Normalized 2D array with shape=target_shape and dtype=dtype
    """
    target_h, target_w = target_shape

    # Convert to grayscale if needed
    if filtered.ndim == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

    # Resize if shape doesn't match
    if filtered.shape != (target_h, target_w):
        filtered = cv2.resize(
            filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Normalize dtype
    if filtered.dtype != dtype:
        if dtype == np.uint8:
            # Normalize to 0-255 range
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        else:
            filtered = filtered.astype(dtype)

    return filtered


def apply_filters_safe(img, filter_functions, filter_names=None):
    """
    Apply multiple filters and return as dictionary with normalized outputs.

    Args:
        img: Input RGB image
        filter_functions: Dict of {name: callable} or list of callables
        filter_names: Optional list of names (if filter_functions is a list)

    Returns:
        Dict of {filter_name: normalized_output}
    """
    target_shape = img.shape[:2]

    if isinstance(filter_functions, dict):
        filters_dict = filter_functions
    elif isinstance(filter_functions, list):
        if filter_names is None:
            filter_names = [f"filter_{i}" for i in range(len(filter_functions))]
        filters_dict = dict(zip(filter_names, filter_functions))
    else:
        raise ValueError("filter_functions must be dict or list")

    results = {}
    for name, func in filters_dict.items():
        try:
            filtered = func()
            normalized = normalize_filter_output(filtered, target_shape)
            results[name] = normalized
        except Exception as e:
            p("Warning", f"Filter '{name}' failed: {e}", color1=c.ORANGE)
            # Return zeros as fallback
            results[name] = np.zeros(target_shape, dtype=np.uint8)

    return results


def create_multichannel_image(filter_outputs, channel_names=None, num_channels=3):
    """
    Create multi-channel image from filter outputs.

    Args:
        filter_outputs: Dict of {name: 2D array} or list of 2D arrays
        channel_names: List of channel names to use (in order)
        num_channels: Number of channels in output (default: 3)

    Returns:
        Multi-channel image with shape (H, W, num_channels)
    """
    if isinstance(filter_outputs, dict):
        if channel_names is None:
            # Use first num_channels filters
            channel_names = list(filter_outputs.keys())[:num_channels]
        channels = [filter_outputs[name] for name in channel_names]
    elif isinstance(filter_outputs, (list, tuple)):
        channels = list(filter_outputs)[:num_channels]
    else:
        raise ValueError("filter_outputs must be dict or list")

    # Pad with last channel if needed
    while len(channels) < num_channels:
        channels.append(channels[-1].copy())

    # Verify all channels have same shape
    shapes = [ch.shape for ch in channels[:num_channels]]
    if len(set(shapes)) != 1:
        raise ValueError(f"Channel shape mismatch: {shapes}")

    # Stack
    multichannel = np.stack(channels[:num_channels], axis=2)
    return multichannel


def create_enhanced_image_robust(img, filter_names, filter_registry=None):
    """
    Robust version of create_enhanced_image with automatic error handling.

    Args:
        img: Input RGB image
        filter_names: List of filter names to apply
        filter_registry: Optional dict of {name: callable}. If None, uses default filters.

    Returns:
        3-channel enhanced image
    """
    from src.exploration.enhancement import clahe_enhance, to_gray
    from src.exploration.filters import cv2_apply_laplacian, cv2_apply_sobel

    gray = to_gray(img)

    # Default filter registry
    if filter_registry is None:
        filter_registry = {
            "laplacian": lambda: cv2_apply_laplacian(img),
            "sobel": lambda: cv2_apply_sobel(img),
            "clahe": lambda: clahe_enhance(gray),
        }

    # Apply filters safely
    filter_outputs = apply_filters_safe(img, filter_registry)

    # Create 3-channel image
    enhanced = create_multichannel_image(
        filter_outputs, channel_names=filter_names, num_channels=3
    )

    return enhanced


# Convenience function for quick testing
def test_filter_pipeline(img_path, filter_names=["laplacian", "sobel", "clahe"]):
    """
    Test the complete filter pipeline on an image.

    Args:
        img_path: Path to image file
        filter_names: List of filters to apply

    Returns:
        Tuple of (original_img, enhanced_img, filter_outputs_dict)
    """
    import cv2

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply filters
    enhanced = create_enhanced_image_robust(img, filter_names)

    # Get individual filter outputs for inspection
    from src.exploration.enhancement import clahe_enhance, to_gray
    from src.exploration.filters import cv2_apply_laplacian, cv2_apply_sobel

    gray = to_gray(img)
    filter_registry = {
        "laplacian": lambda: cv2_apply_laplacian(img),
        "sobel": lambda: cv2_apply_sobel(img),
        "clahe": lambda: clahe_enhance(gray),
    }

    filter_outputs = apply_filters_safe(img, filter_registry)

    return img, enhanced, filter_outputs


# From C:\github\Tree-Canopy-Detection\src\exploration\kernels.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.exploration.visualize import show_side_by_side


def laplacian_kernel() -> np.ndarray:
    """
    Standard 3x3 Laplacian kernel.
    """
    return np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )


def visualize_kernel(kernel, title="Kernel", cmap=None):
    """
    Visualize a 2D convolution kernel as an image and 3D surface.
    """

    # Auto-select colormap if not provided
    if cmap is None:
        if np.any(kernel < 0):
            cmap = "seismic"  # diverging: shows negative vs positive clearly
        else:
            cmap = "gray"  # sequential: good for blur kernels

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.patch.set_facecolor("white")

    # 2D heatmap
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.imshow(kernel, cmap=cmap)
    ax2d.set_title(f"{title.replace('_', ' ')} (2D heatmap)", fontsize=8)
    ax2d.axis("off")

    # 3D surface
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    x = np.arange(kernel.shape[1])
    y = np.arange(kernel.shape[0])
    X, Y = np.meshgrid(x, y)
    ax3d.plot_surface(X, Y, kernel, cmap=cmap, edgecolor="k")
    ax3d.set_title(f"{title.replace('_', ' ')} (3D surface)", fontsize=8)

    plt.show()


def apply_kernel_using_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a kernel to a grayscale image using convolution.
    """
    from scipy.signal import convolve2d

    if image.ndim == 3:
        raise ValueError("apply_kernel expects a single channel image")

    out = convolve2d(image, kernel, mode="same", boundary="symm")
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_custom_kernel(image, kernel, show_image=False):
    """
    Apply a custom kernel to an image and visualize the response.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered = cv2.filter2D(gray, -1, kernel)

    if show_image:
        show_side_by_side(
            gray,
            filtered,
            titles=["Original", "Kernel Response"],
            cmaps=["gray", "gray"],
        )

    return filtered


def get_kernels(name: str = "all"):
    """
    Return a dictionary of named standard kernels for exploration.
    """
    kernels = {
        "Identity": np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Sobel_X": np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        ),
        "Sobel_Y": np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=np.float32,
        ),
        "Prewitt_X": np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        ),
        "Prewitt_Y": np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.float32,
        ),
        "Scharr_X": np.array(
            [
                [-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3],
            ],
            dtype=np.float32,
        ),
        "Scharr_Y": np.array(
            [
                [-3, -10, -3],
                [0, 0, 0],
                [3, 10, 3],
            ],
            dtype=np.float32,
        ),
        "Roberts_X": np.array(
            [
                [1, 0],
                [0, -1],
            ],
            dtype=np.float32,
        ),
        "Roberts_Y": np.array(
            [
                [0, 1],
                [-1, 0],
            ],
            dtype=np.float32,
        ),
        "Laplacian_3x3": np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        ),
        "Laplacian_5x5": np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Box_Blur_3x3": np.ones((3, 3), dtype=np.float32) / 9,
        "Box_Blur_5x5": np.ones((5, 5), dtype=np.float32) / 25,
        "cv2_Gaussian_3x3": cv2.getGaussianKernel(3, 1) @ cv2.getGaussianKernel(3, 1).T,
        "cv2_Gaussian_5x5": cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T,
        "Sharpen_Basic": np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        ),
        "High_Boost": np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
        "Emboss_1": np.array(
            [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.float32,
        ),
        "Emboss_2": np.array(
            [
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        ),
        "Edge_Enhance": np.array(
            [
                [0, 0, 0],
                [-1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Edge_Enhance_Strong": np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
        "Motion_Blur_5x5": np.eye(5, dtype=np.float32) / 5,
        "Motion_Blur_9x9": np.eye(9, dtype=np.float32) / 9,
        "Gradient_Magnitude": np.array(
            [
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        ),
        "High_Pass_3x3": np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
    }

    sobel_x = kernels["Sobel_X"]
    sobel_y = kernels["Sobel_Y"]
    kernels["Sobel"] = np.sqrt(sobel_x**2 + sobel_y**2)

    def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Generate a 2D Gaussian kernel.
        """
        k = size // 2
        x = np.arange(-k, k + 1)
        y = np.arange(-k, k + 1)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel.astype(np.float32)

    for i in range(1, 3):
        kernels[f"Gaussian_2x2_sigma{i}"] = gaussian_kernel(size=2, sigma=i)
        kernels[f"Gaussian_3x3_sigma{i}"] = gaussian_kernel(size=3, sigma=i)
        kernels[f"Gaussian_5x5_sigma{i}"] = gaussian_kernel(size=5, sigma=i)
        kernels[f"Gaussian_7x7_sigma{i}"] = gaussian_kernel(size=7, sigma=i)
        kernels[f"Gaussian_9x9_sigma{i}"] = gaussian_kernel(size=9, sigma=i)

    if name == "keys":
        return kernels.keys()

    if name == "all":
        return kernels

    for key in kernels:
        if key.lower() == name.lower():
            return kernels[key]

    raise ValueError(f"Unknown kernel '{name}'")


def display_standard_kernels(cmap="Greens_r"):
    """
    Display all standard kernels from get_standard_kernels as heatmaps.
    """
    kernels = get_kernels()

    n = len(kernels)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    for ax, (name, kernel) in zip(axes, kernels.items()):
        vmax = np.max(np.abs(kernel))
        im = ax.imshow(kernel, cmap=cmap, vmin=-vmax, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        all_ints = np.allclose(kernel, np.round(kernel))
        if not all_ints:
            decimals = []
            for val in kernel.flatten():
                if val != 0:
                    s = f"{val:.8f}".rstrip("0").rstrip(".")
                    if "." in s:
                        decimals.append(len(s.split(".")[1]))
            max_decimals = max(decimals) if decimals else 0
            if max_decimals == 1:
                fmt = "{:.1f}"
            elif max_decimals == 2:
                fmt = "{:.2f}"
            else:
                fmt = "{:.3f}"
        else:
            fmt = "{}"

        for (i, j), val in np.ndenumerate(kernel):
            rgba = cmap_obj(norm(val))
            brightness = np.dot(rgba[:3], [0.299, 0.587, 0.114])
            text_color = "black" if brightness > 0.5 else "white"
            if all_ints or val == 0:
                text_str = f"{int(val)}"
            else:
                text_str = fmt.format(val)

            ax.text(
                j, i, text_str, ha="center", va="center", fontsize=6, color=text_color
            )

        ax.set_title(name.replace("_", " "), fontsize=6)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(kernels) :]:
        ax.axis("off")

    plt.suptitle("Standard Kernels (Values and Structure)", fontsize=8, y=1.02)
    plt.tight_layout()
    plt.show()


def make_motion_kernel(size: int = 9, angle: float = 0.0) -> np.ndarray:
    """
    Create a motion blur kernel of a given size and angle.
    """
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(kernel, (size // 2, 0), (size // 2, size - 1), color=1.0, thickness=1)

    # Rotate the vertical line to desired angle
    center = (size // 2, size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))

    # Normalize so sum = 1
    s = kernel.sum()
    if s != 0:
        kernel /= s
    return kernel


def make_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Create a 2D Gaussian blur kernel.
    """
    if size % 2 == 0:
        size += 1

    gk = cv2.getGaussianKernel(size, sigma)
    kernel = gk @ gk.T
    kernel /= kernel.sum()
    return kernel


def make_directional_edge_kernel(
    size: int = 3, direction: str = "horizontal"
) -> np.ndarray:
    """
    Create a simple directional edge detection kernel.

    """
    if size < 3:
        raise ValueError("size must be at least 3")

    base = np.zeros((size, size), dtype=float)

    if direction == "horizontal":
        base[size // 2, :] = np.linspace(-1, 1, size)
    elif direction == "vertical":
        base[:, size // 2] = np.linspace(-1, 1, size)
    elif direction == "diag_pos":
        np.fill_diagonal(base[:, ::-1], np.linspace(-1, 1, size))
    elif direction == "diag_neg":
        np.fill_diagonal(base, np.linspace(-1, 1, size))
    else:
        raise ValueError(
            "direction must be one of: horizontal, vertical, diag_pos, diag_neg"
        )

    denom = np.sum(np.abs(base))
    if denom != 0:
        base /= denom
    return base


# From C:\github\Tree-Canopy-Detection\src\exploration\visualize.py
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_side_by_side(
    *images: np.ndarray,
    titles: Optional[Tuple[str, ...]] = None,
    cmaps: Optional[Tuple[str, ...]] = None,
    maxcolumns: Optional[int] = None,
    preserve_values: bool = False,
    vmax=None,
    kernel=None,
    kernel_title=None,
    kernel_cmap="seismic",
    mask_colors=None,
    figsize=4,
) -> None:
    """Show multiple images side by side."""

    count = len(images)
    # auto adjust long grids
    if maxcolumns is not None and maxcolumns > 10:
        titles = tuple(str(i) for i in range(count))
        scale = max(1.0, maxcolumns / 10)
        title_fontsize = min(22, 20 * scale)
    else:
        title_fontsize = 10

    # Default titles
    if titles is None:
        # default = ["Image", "Mask", "Overlay"]
        # titles = default[:count]
        titles = tuple(f"Image {i + 1}" for i in range(count))

    if cmaps is None:
        cmaps = tuple([None] * count)

    # ---- grid layout ----
    if maxcolumns is None or maxcolumns >= count:
        ncols = count
        nrows = 1
    else:
        ncols = maxcolumns
        nrows = int(np.ceil(count / maxcolumns))

    add_kernel = kernel is not None
    if add_kernel:
        ncols = ncols + 2  # reserve 2 columns for 2D + 3D kernel plots

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))
    fig.patch.set_facecolor("white")

    # Flatten axes for easy iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate over images
    for ax, img, title, cmap in zip(axes[:count], images, titles, cmaps):
        # ---------------------------------------------------
        # color-mask logic FIRST and ALWAYS before type checks
        # ---------------------------------------------------
        # if "mask" in title.lower() and isinstance(img, tuple) and len(img) == 2:
        if mask_colors and isinstance(img, tuple) and len(img) == 2:

            mask, cls = img
            # choose color
            # if mask_colors and cls in mask_colors:
            if cls in mask_colors:
                color = tuple(mask_colors[cls])
            else:
                color = (255, 255, 255)

            # convert single-channel mask to 0/255
            if mask.dtype != np.uint8:
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask_u8 = mask * 255 if mask.max() <= 1 else mask

            # create rgb mask
            img = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
            img[mask_u8 > 0] = color

        # ---------------------------------------------------
        # now handle normal ndarray inputs
        # ---------------------------------------------------
        elif isinstance(img, np.ndarray):
            # Handle raw image arrays
            if not preserve_values:
                # Handle [0,1] range images properly
                if img.max() <= 1.0:
                    # Image is in [0,1] range - scale to [0,255]
                    img = (img * 255).astype(np.uint8)
                else:
                    # Image is in larger range - clip to [0,255]
                    img = np.clip(img, 0, 255).astype(np.uint8)
            # else: keep raw values (signed floats/ints)

        # ---------------------------------------------------
        # matplotlib figure
        # ---------------------------------------------------
        elif hasattr(img, "canvas"):
            # render the source figure into an image
            img.canvas.draw()

            # get RGBA buffer from source figure
            buf = np.asarray(img.canvas.buffer_rgba())

            # convert RGBA to RGB by dropping alpha
            img = buf[:, :, :3]

        # ---------------------------------------------------
        # unsupported type
        # ---------------------------------------------------
        else:
            raise TypeError(
                f"Unsupported type {type(img)} passed to show_side_by_side. "
                "Expected numpy.ndarray, (mask,class_name) tuple, or matplotlib Figure."
            )

        # Visualization
        # if title == "Mask":
        #     ax.imshow(img, "gray")
        # if "mask" in title.lower():
        if "mask" in title.lower() and isinstance(img, np.ndarray) and img.ndim == 2:
            ax.imshow(img, "gray")
        elif cmap is not None:
            # If preserve_values is True, we need symmetric vmin/vmax
            if preserve_values and np.issubdtype(img.dtype, np.number):
                # convert before passing
                # lap_img = apply_kernel(img_gray, lk).astype(np.float32)
                if vmax is None:
                    vmax = float(np.max(np.abs(img)))

                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)

        ax.set_facecolor("white")
        if maxcolumns is not None and maxcolumns > 10:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
        ax.axis("off")

    # if add_kernel:
    #     for spine in ax2d.spines.values():
    #         spine.set_visible(True)
    #         spine.set_edgecolor("black")
    #         spine.set_linewidth(2)

    # remove empty axes
    for ax in axes[count:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def show_image(
    image: np.ndarray, title: str = "", return_img: bool = False, cmap="gray"
):
    """
    Show an image using matplotlib.
    """
    # Convert float images safely - handle [0,1] range properly
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Image is in [0,1] range - convert to [0,255]
            img = (image * 255).astype(np.uint8)
        else:
            # Image is in larger range - clip to [0,255]
            img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    if return_img:
        return img
    else:
        plt.figure(figsize=(5, 5))
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()


# using from PIL import Image to be able to show pure white and black
def show_mask(mask: np.ndarray, title: str = "", return_img: bool = False, cmap="gray"):
    """
    Show a mask with proper handling of class indices.
    """
    # Handle multi-class masks (class indices 0,1,2)
    if mask.dtype != np.uint8:
        if mask.max() <= 2:
            # Multi-class mask with indices [0,1,2] - scale to visible range
            mask_img = (mask * 127).astype(np.uint8)  # 0->0, 1->127, 2->254
        else:
            # Binary mask [0,1] - convert to [0,255]
            mask_img = (mask > 0.5).astype(np.uint8) * 255
    else:
        if mask.max() <= 2:
            # Already uint8 with class indices [0,1,2]
            mask_img = mask * 127
        elif mask.max() <= 1:
            # Binary mask [0,1] as uint8
            mask_img = mask * 255
        else:
            # Already in proper range
            mask_img = mask

    if return_img:
        return mask_img

    # use PIL for exact grayscale
    from PIL import Image

    img = Image.fromarray(mask_img, mode="L")

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    title: str = "",
    return_img: bool = False,
):
    """
    Show an image with a red mask overlay.
    """
    # Convert image to uint8
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    # Handle mask properly - don't multiply class indices by 255
    if mask.max() <= 2:
        # Multi-class mask [0,1,2] - convert to binary [0,1] then to [0,255]
        mask_binary = (mask > 0).astype(np.float32)  # Any class > 0 = tree
        mask_u8 = (mask_binary * 255).astype(np.uint8)
    else:
        # Already processed mask
        mask_u8 = mask.astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8  # Red channel for trees

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)

    if return_img:
        return overlay
    else:
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()


def show_stages(stages: dict, cmaps=None, maxcolumns=5) -> None:
    """
    Display multiple enhancement or filtering stages.
    Keys must be stage names. Values are images.
    """
    images = list(stages.values())
    titles = list(stages.keys())
    if cmaps is None:
        cmaps = ["gray"] * len(images)

    show_side_by_side(*images, titles=titles, cmaps=cmaps)

    # n = len(stages)
    # cols = min(4, n)
    # rows = int(np.ceil(n / cols))
    #
    # plt.figure(figsize = (4 * cols, 4 * rows))
    #
    # for i, (name, img) in enumerate(stages.items(), 1):
    #     plt.subplot(rows, cols, i)
    #     if img.ndim == 2:
    #         plt.imshow(img, cmap = cmap)
    #     else:
    #         plt.imshow(img)
    #     plt.title(name)
    #     plt.axis("off")
    #
    # plt.tight_layout()
    # plt.show()


# From C:\github\Tree-Canopy-Detection\src\exploration\__init__.py


