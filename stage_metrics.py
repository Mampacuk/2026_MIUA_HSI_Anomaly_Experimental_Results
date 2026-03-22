import os
from typing import Any

import numpy as np
from metrics import compute_auc, compute_ser, compute_aer


def minmax_normalize(pred):
    pmin = np.min(pred)
    pmax = np.max(pred)

    if pmax - pmin == 0:
        return np.zeros_like(pred)

    return (pred - pmin) / (pmax - pmin)

def format_metric(name: str, value: float | np.floating[Any], ndigits: int | None=None):
    return f"{name}: {value}" + (f" ({ndigits}-rounded: {round(value, ndigits)})" if ndigits else "")

def format_metric_line(prefix: str | None, auc_val: float | np.floating[Any], ser_val: float | np.floating[Any],
                       aer_val: float | np.floating[Any], *, suffix: str="", ndigits: int | None=None):
    line = prefix + ". " if prefix else ""
    line += (f"{format_metric("AUC", auc_val, ndigits)}, {format_metric("SER", ser_val, ndigits)}, "
             f"{format_metric("AER", aer_val, ndigits)}")
    return line + suffix + "\n"

def compute_metrics(eval_data, dataset_to_group, metrics_folder, *, ndigits=None):

    os.makedirs(metrics_folder, exist_ok=True)

    scores = {}
    clamps = {}

    for (method, dataset), item in eval_data.items():
        pred = item["pred"]
        gt = item["gt"]

        pred_norm = minmax_normalize(pred)

        auc_val, _, _ = compute_auc(gt, pred_norm)
        ser = compute_ser(gt, pred_norm)
        aer = compute_aer(gt, pred_norm)

        scores.setdefault(method, {})[dataset] = (auc_val, ser, aer)
        clamps.setdefault(method, {})[dataset] = item["clamp_percentile"]

    groups = set(dataset_to_group.values())

    for method in scores:

        for group in groups:

            out_path = os.path.join(metrics_folder, f"{method}_{group}.txt")

            aucs, sers, aers = [], [], []

            with open(out_path, "w") as f:

                for dataset in scores[method]:

                    if dataset_to_group[dataset] != group:
                        continue

                    auc_val, ser_val, aer_val = scores[method][dataset]

                    aucs.append(auc_val)
                    sers.append(ser_val)
                    aers.append(aer_val)

                    f.write(format_metric_line(dataset, auc_val, ser_val, aer_val, ndigits=ndigits,
                                               suffix=f" (clamped at {clamps[method][dataset]} prctl)"
                                                      if clamps[method][dataset] else ""))

                if aucs:
                    f.write(format_metric_line(f"average of {len(aucs)} methods", np.mean(aucs), np.mean(sers),
                                               np.mean(aers), ndigits=ndigits))

        # overall file for a method
        out_path = os.path.join(metrics_folder, f"{method}.txt")

        aucs, sers, aers = [], [], []

        with open(out_path, "w") as f:

            for dataset in scores[method]:

                auc_val, ser_val, aer_val = scores[method][dataset]

                aucs.append(auc_val)
                sers.append(ser_val)
                aers.append(aer_val)

                f.write(format_metric_line(dataset, auc_val, ser_val, aer_val, ndigits=ndigits,
                                           suffix=f" (clamped at {clamps[method][dataset]} prctl)"
                                           if clamps[method][dataset] else ""))

            if aucs:
                f.write(format_metric_line(f"average of {len(aucs)} methods", np.mean(aucs), np.mean(sers),
                                           np.mean(aers), ndigits=ndigits))