import numpy as np
import torch


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
    Compute IoU, Dice, Accuracy for binary segmentation.
    """
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
        pred = torch.nn.functional.interpolate(
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
    # Handle SegFormer output format
    if hasattr(pred, "logits"):
        pred = pred.logits

    # Convert to tensor if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)

    # Get target spatial size
    if isinstance(true, torch.Tensor):
        target_size = true.shape[-2:]
    else:
        target_size = true.shape[-2:]

    # Resize prediction to match target size if needed
    if pred.shape[-2:] != target_size:
        pred = torch.nn.functional.interpolate(
            pred, size=target_size, mode="bilinear", align_corners=False
        )

    # BINARY MODE: If pred has 1 channel, use binary metrics
    if pred.shape[1] == 1:
        # Squeeze channel dimension from true if present
        if isinstance(true, torch.Tensor) and true.ndim == 4 and true.shape[1] == 1:
            true_binary = true.squeeze(1)  # [B, H, W]
        else:
            true_binary = true

        # Apply sigmoid and threshold
        pred_binary = torch.sigmoid(pred).squeeze(1) > 0.5  # [B, H, W]

        if isinstance(true_binary, torch.Tensor):
            true_binary = true_binary > 0.5

            # Convert to numpy
            pred_np = pred_binary.detach().cpu().numpy()
            true_np = true_binary.detach().cpu().numpy()
        else:
            pred_np = pred_binary.detach().cpu().numpy()
            true_np = true_binary > 0.5

        # Compute binary metrics
        tp = (pred_np & true_np).sum()
        fp = (pred_np & ~true_np).sum()
        fn = (~pred_np & true_np).sum()
        tn = (~pred_np & ~true_np).sum()

        inter = float(tp)
        union = float(tp + fp + fn)

        return {
            "iou": inter / union if union > 0 else 0,
            "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
        }

    # MULTI-CLASS MODE: If pred > 1 channels
    # Convert logits to class predictions
    if isinstance(pred, torch.Tensor):
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred_classes = torch.argmax(pred, dim=1).detach().cpu().numpy()  # [B, H, W]
    else:
        pred_classes = np.argmax(pred, axis=1)

    if isinstance(true, torch.Tensor):
        # Squeeze channel dimension if present
        if true.ndim == 4 and true.shape[1] == 1:
            true = true.squeeze(1)
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
        ious[f"iou_{class_names.get(cls_id, f"class_{cls_id}")}"] = iou

    # Mean IoU (excluding background class 0)
    if num_classes > 1:
        tree_ious = [ious[f"iou_{class_names[i]}"] for i in range(1, num_classes)]
        ious["mean_iou"] = sum(tree_ious) / len(tree_ious)
    else:
        ious["mean_iou"] = ious.get("iou_background", 0.0)

    # Overall pixel accuracy
    correct = (pred_classes == true_classes).sum()
    total = pred_classes.size
    ious["acc"] = float(correct) / float(total)

    # Aliases for compatibility
    ious["iou"] = ious["mean_iou"]
    ious["dice"] = 0.0  # Placeholder

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for cls_id in range(1, num_classes):  # Skip background class 0
        pred_mask = pred_classes == cls_id
        true_mask = true_classes == cls_id

        tp = np.logical_and(pred_mask, true_mask).sum()
        fp = np.logical_and(pred_mask, ~true_mask).sum()
        fn = np.logical_and(~pred_mask, true_mask).sum()

        all_tp += tp
        all_fp += fp
        all_fn += fn

    # Overall precision and recall
    ious["precision"] = float(all_tp) / float(all_tp + all_fp + 1e-8)
    ious["recall"] = float(all_tp) / float(all_tp + all_fn + 1e-8)

    # F1 score
    prec = ious["precision"]
    rec = ious["recall"]
    ious["f1_score"] = 2 * (prec * rec) / (prec + rec + 1e-8)

    return ious
