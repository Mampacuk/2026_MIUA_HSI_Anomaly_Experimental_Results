import os

import numpy as np
import scipy.io as sio
from PIL import Image
from skimage.transform import resize

from diagnostics import inspect_score_distribution


def discover_detector_files(methods, datasets, groups, det_folder):
    """
    Returns
    -------
    detector_files : dict
        {method: {dataset: filename}}
    dataset_to_group : dict
        {dataset: group}
    """

    detector_files = {m: {} for m in methods}

    # -------------------------
    # Dataset -> group mapping
    # -------------------------

    dataset_to_group = {}

    for dataset in datasets:

        matched = [g for g in groups if dataset.startswith(g)]

        if len(matched) > 1:
            raise ValueError(f"{dataset} matches multiple groups")
        if len(matched) == 0:
            raise ValueError(f"{dataset} does not match any group")

        dataset_to_group[dataset] = matched[0]

    # -------------------------
    # Scan method folders
    # -------------------------

    for method in methods:

        method_folder = os.path.join(det_folder, method)

        if not os.path.isdir(method_folder):
            raise RuntimeError(
                f"Expected method folder not found: {method_folder}"
            )

        prefix = method + "_"

        for f in os.listdir(str(method_folder)):

            if not f.endswith(".mat"):
                continue

            if not f.startswith(prefix):
                continue

            remainder = f[len(prefix):]

            for dataset in datasets:

                if remainder.startswith(dataset):

                    if dataset in detector_files[method]:
                        raise RuntimeError(
                            f"Multiple detector files found for "
                            f"{method} / {dataset}: "
                            f"{detector_files[method][dataset]} and {f}"
                        )

                    detector_files[method][dataset] = f

    return detector_files, dataset_to_group


def preload_evaluation_data(detector_files, gt_folder, det_folder, *,
                            clamp_config=None,
                            histogram_debug=False):
    eval_data = {}

    for method in detector_files:

        for dataset in detector_files[method]:
            filename = detector_files[method][dataset]

            pred = sio.loadmat(
                os.path.join(det_folder, method, filename)
            )["show"].astype(float)

            # -------------------------
            # Optional clamping
            # -------------------------

            clamp_percentile = None

            if clamp_config and (method, dataset) in clamp_config:
                clamp_percentile = clamp_config[(method, dataset)]
                threshold = np.percentile(pred, clamp_percentile * 100)
                pred = np.minimum(pred, threshold)

            # -------------------------
            # Histogram diagnostics
            # -------------------------

            if histogram_debug:
                inspect_score_distribution(
                    pred,
                    method,
                    dataset
                )

            # -------------------------
            # Load GT
            # -------------------------

            gt = np.array(
                Image.open(os.path.join(gt_folder, f"{dataset}.png"))
            )
            gt = gt > 0
            gt = align_ground_truth(gt, pred.shape)

            eval_data[(method, dataset)] = {
                "file": filename,
                "pred": pred,
                "gt": gt,
                "clamp_percentile": clamp_percentile
            }

    return eval_data


def align_ground_truth(gt, target_shape):
    gh, gw = gt.shape
    th, tw = target_shape

    if (gh, gw) == (th, tw):
        return gt

    scale_h = th / gh
    scale_w = tw / gw

    # ensure isotropic scaling
    if abs(scale_h - scale_w) > 1e-6:
        raise RuntimeError(
            f"Incompatible GT scaling {gt.shape} -> {target_shape}"
        )

    scale = scale_h

    # verify scale is a power of two (allowing fractions like 0.5, 0.25)
    log2_scale = np.log2(scale)

    if abs(log2_scale - round(log2_scale)) > 1e-6:
        raise RuntimeError(
            f"Scale {scale} between {gt.shape} and {target_shape} "
            f"is not a power of two"
        )
    resized = resize(
        gt.astype(float),
        target_shape,
        order=0,  # nearest-neighbor to preserve binary nature
        preserve_range=True,
        anti_aliasing=False
    ) > 0.5  # threshold back to boolean
    assert resized.shape == target_shape
    return resized
