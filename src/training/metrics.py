import numpy as np
import torch


def _to_numpy( pred: torch.Tensor, true: torch.Tensor ):
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


def compute_confusion( pred_bin: np.ndarray, true_bin: np.ndarray ):
    """
    Compute basic confusion counts.
    """
    tp = np.logical_and(pred_bin == 1, true_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, true_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, true_bin == 0).sum()
    return tp, fp, fn, tn


def compute_metrics( pred: torch.Tensor, true: torch.Tensor ):
    """
    Compute IoU, Dice, Accuracy for one batch of predictions.
    Accepts torch tensors. Returns dict of floats.
    """
    pred_bin, true_bin = _to_numpy(pred, true)
    tp, fp, fn, tn = compute_confusion(pred_bin, true_bin)

    inter = float(tp)
    union = float(tp + fp + fn)
    iou = inter / union if union > 0 else 0.0

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
    }
