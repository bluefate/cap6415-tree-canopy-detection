import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(pred: torch.Tensor, true: torch.Tensor):
    """
    Convert prediction and ground truth tensors to numpy arrays.
    Applies sigmoid to prediction if needed, then thresholds at 0.5.
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8)
    true_bin = (true > 0.5).astype(np.uint8)
    return pred_bin, true_bin


def compute_confusion(pred_bin: np.ndarray, true_bin: np.ndarray):
    """
    Compute basic confusion counts.
    """
    tp = np.logical_and(pred_bin == 1, true_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, true_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, true_bin == 0).sum()
    return tp, fp, fn, tn


def compute_metrics(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute IoU, Dice, Accuracy for one batch of predictions.
    Accepts torch tensors. Returns dict of floats.
    Handles special output formats if needed (SegFormer) and spatial size mismatches.
    """

    ## Replaced current compute_metrics:
    # pred_bin, true_bin = _to_numpy(pred, true)
    # tp, fp, fn, tn = compute_confusion(pred_bin, true_bin)
    #
    # inter = float(tp)
    # union = float(tp + fp + fn)
    # iou = inter / union if union > 0 else 0.0
    #
    # dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    # acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    # prec = tp / (tp + fp + 1e-8)
    # rec = tp / (tp + fn + 1e-8)
    #
    # return {
    #     "iou": float(iou),
    #     "dice": float(dice),
    #     "acc": float(acc),
    #     "precision": float(prec),
    #     "recall": float(rec),
    # }

    # Handle SegFormer output format
    if hasattr(pred, "logits"):
        pred = pred.logits

    # Convert to tensor if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)

    # Get target spatial size
    if isinstance(true, torch.Tensor):
        target_size = true.shape[-2:]  # (H, W)
    else:
        target_size = true.shape[-2:]

    # Resize prediction to match target size if needed
    if pred.shape[-2:] != target_size:
        pred = F.interpolate(
            pred, size=target_size, mode="bilinear", align_corners=False
        )

    # Apply sigmoid for binary segmentation
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).detach().cpu().numpy()

    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_bin = pred > 0.5
    true_bin = true > 0.5

    tp = (pred_bin & true_bin).sum()
    fp = (pred_bin & ~true_bin).sum()
    fn = (~pred_bin & true_bin).sum()
    tn = (~pred_bin & ~true_bin).sum()

    inter = float(tp)
    union = float(tp + fp + fn)

    return {
        "iou": inter / union if union > 0 else 0,
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
        "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def compute_metrics_multiclass(
    pred: torch.Tensor, true: torch.Tensor, num_classes: int = 3
):
    """
    Compute per-class IoU and mean IoU for multi-class segmentation.
    """
    # Convert logits to class predictions
    if isinstance(pred, torch.Tensor):
        pred_classes = torch.argmax(pred, dim=1).detach().cpu().numpy()  # [B, H, W]
    else:
        pred_classes = np.argmax(pred, axis=1)

    if isinstance(true, torch.Tensor):
        true_classes = true.detach().cpu().numpy()  # [B, H, W]
    else:
        true_classes = true

    # Compute IoU for each class
    ious = {}
    class_names = {0: "background", 1: "individual_tree", 2: "group_of_trees"}

    for cls_id in range(num_classes):
        pred_mask = pred_classes == cls_id
        true_mask = true_classes == cls_id

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        iou = float(intersection) / float(union + 1e-8)
        ious[f"iou_{class_names[cls_id]}"] = iou

    # Mean IoU (excluding background class 0)
    tree_ious = [ious["iou_individual_tree"], ious["iou_group_of_trees"]]
    ious["mean_iou"] = sum(tree_ious) / len(tree_ious)

    # Also compute overall pixel accuracy
    correct = (pred_classes == true_classes).sum()
    total = pred_classes.size
    ious["acc"] = float(correct) / float(total)

    # For compatibility with existing code, also add these
    ious["iou"] = ious["mean_iou"]  # Alias
    ious["dice"] = 0.0  # Placeholder - can compute per-class dice if needed

    return ious
