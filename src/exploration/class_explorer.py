from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data.annotations import AnnotationEntry
from src.data.image_loader import load_image
from src.exploration.visualize import (
    show_side_by_side,
)
from src.utils.config import Config
from src.utils.helpers import p, t


# -----------------------------------------------------------
# Core helpers
# -----------------------------------------------------------
config = Config.load()
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
        marker = " 🏆" if i == best_iou_idx else ""
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
