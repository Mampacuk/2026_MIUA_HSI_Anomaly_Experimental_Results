import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_auc(gt, pred):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    fpr, tpr, _ = roc_curve(gt_flat, pred_flat)
    return auc(fpr, tpr), fpr, tpr


def compute_ser(gt, pred):
    """
    Square Error Ratio (SER)
    Equivalent to mean Brier score over pixels.
    Lower is better. Range: [0,1].
    """
    eps = 1e-6
    assert np.min(pred) >= -eps and np.max(pred) <= 1 + eps, (
        f"Prediction values must lie in [0,1]. "
        f"Found range [{pred.min()}, {pred.max()}]"
    )
    gt = gt.astype(float)
    return np.mean((pred - gt) ** 2)


def compute_aer(gt, pred):
    """
    Area Error Ratio based on modified ROC curves.
    """

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    fpr, tpr, thresholds = roc_curve(gt_flat, pred_flat)
    if np.isinf(thresholds[0]):
        fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    pmin = np.min(pred_flat)
    pmax = np.max(pred_flat)

    if pmax - pmin == 0:
        return np.nan

    # normalize thresholds
    normalized_thresholds = (thresholds - pmin) / (pmax - pmin)

    # ensure increasing order for AUC integration
    order = np.argsort(normalized_thresholds)

    normalized_thresholds = normalized_thresholds[order]
    fpr = fpr[order]
    tpr = tpr[order]

    a_fpr = auc(normalized_thresholds, fpr)
    a_tpr = auc(normalized_thresholds, tpr)

    if 1 - a_tpr == 0:
        return np.inf

    aer = (1 - a_fpr) / (1 - a_tpr)

    return aer