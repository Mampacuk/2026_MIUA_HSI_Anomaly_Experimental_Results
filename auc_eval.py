#!/usr/bin/env python3
"""
Generalized evaluator for detector .mat outputs.

Filename convention expected (customizable at top):
    {METHOD_NAME}_{datasetname}_{arg1}{value1}_{arg2}{value2}.mat

Example for LRX:
    LRX_bovine_in5_out13.mat

Configure:
    METHOD_NAME = "LRX"
    ARG_LIST = [("in", int), ("out", int)]
Order in ARG_LIST is significant.
Floats may be encoded using 'p' instead of '.' (e.g. lr0p01 -> 0.01).
"""

import os
import re
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

# --- Configure here ---
RESULTS_DIR = "detector_outputs/PTA_best"
GT_DIR = "detectors/input130x174"
METHOD_NAME = "PTA_best"
# list of (argument_string_in_filename, type). order matters.
ARG_LIST = []
# ------------------------

from metrics import compute_auc  # your existing function


def _type_regex(pytype):
    """Return a regex snippet that captures the string for this type."""
    if pytype is int:
        return r"(\d+)"
    elif pytype is float:
        # accept digits like 12, 12.34 or 12p34 (p used instead of dot)
        return r"(\d+(?:[\.p]\d+)?)"
    else:
        # fallback: capture non-underscore sequence
        return r"([^_]+)"


def _convert_value(s: str, pytype):
    """Convert the captured string to the desired python type.
    Supports 'p' as decimal point for floats.
    """
    if pytype is int:
        return int(s)
    if pytype is float:
        s2 = s.replace("p", ".")
        return float(s2)
    # fallback
    return pytype(s)


def build_filename_pattern(method_name, arg_list):
    """Builds and compiles a regex to parse filenames for:
       methodname_dataset_arg1{val}_arg2{val}.mat
       Returns compiled regex with a capturing group for dataset (group 1)
       followed by groups for each argument in order.
    """
    parts = []
    esc_method = re.escape(method_name)
    parts.append(rf"^{esc_method}_")           # start with METHOD_
    parts.append(r"(.+?)")                     # dataset name (non-greedy)
    # for each arg add '_' + argname + <capture>
    for arg_name, arg_type in arg_list:
        parts.append(rf"_{re.escape(arg_name)}" + _type_regex(arg_type))
    parts.append(r"\.mat$")                    # extension
    pattern = "".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE)


def parse_filename(filename, compiled_pattern, arg_list):
    """Parse filename using compiled_pattern.
       Returns (dataset_name, args_dict) or (None, None) if not matched.
    """
    m = compiled_pattern.match(filename)
    if not m:
        return None, None
    # group 1 is dataset, subsequent groups correspond to args in order
    dataset = m.group(1)
    args = {}
    # groups start from index 2 for args
    for i, (arg_name, arg_type) in enumerate(arg_list, start=2):
        raw = m.group(i)
        try:
            val = _convert_value(raw, arg_type)
        except Exception:
            # conversion failed; return None to indicate mismatch
            return None, None
        args[arg_name] = val
    return dataset, args


def find_gt_for_dataset(gt_dir, dataset_name):
    """Find and return the GT 'map' for dataset_name in gt_dir.
       Heuristic: returns first file containing dataset_name as substring and having a 'map' variable.
       Raises RuntimeError if no GT found.
    """
    # try exact-ish matches first (case-insensitive)
    candidates = [f for f in os.listdir(gt_dir) if f.lower().endswith(".mat")]
    # prefer files where the dataset_name is a separate token (surrounded by underscores or start/end)
    dataset_lower = dataset_name.lower()
    prioritized = []
    fallback = []
    for f in candidates:
        fl = f.lower()
        if dataset_lower in fl:
            # check token boundaries
            if re.search(rf'(^|_){re.escape(dataset_lower)}(_|\.|$)', fl):
                prioritized.append(f)
            else:
                fallback.append(f)
    search_list = prioritized + fallback + candidates
    seen = set()
    for f in search_list:
        if f in seen:
            continue
        seen.add(f)
        path = os.path.join(gt_dir, f)
        try:
            data = loadmat(path)
        except Exception:
            continue
        if "map" in data:
            return data["map"]
    raise RuntimeError(f"No GT (.mat with variable 'map') found for dataset '{dataset_name}' in {gt_dir}")


def normalize_auc_value(auc_raw):
    """Normalize compute_auc output to a float.
       Accepts: float, numpy scalar, tuple/list (take first element), etc.
    """
    if isinstance(auc_raw, (list, tuple)):
        candidate = auc_raw[0]
    else:
        candidate = auc_raw
    # If it's numpy scalar, convert to python float
    try:
        return float(np.asarray(candidate).item())
    except Exception:
        # last resort
        return float(candidate)


def main(results_dir, gt_dir):
    compiled_pat = build_filename_pattern(METHOD_NAME, ARG_LIST)
    # cache GT maps per dataset
    gt_cache = {}
    scores = defaultdict(list)

    for fname in sorted(os.listdir(results_dir)):
        if not fname.lower().endswith(".mat"):
            continue
        dataset, args = parse_filename(fname, compiled_pat, ARG_LIST)
        if dataset is None:
            # filename didn't match the configured pattern
            continue

        fullpath = os.path.join(results_dir, fname)
        try:
            data = loadmat(fullpath)
        except Exception as ex:
            print(f"Skipping {fname} (failed to load .mat): {ex}")
            continue

        if "show" not in data:
            print(f"Skipping {fname} (no 'show' variable)")
            continue

        pred = data["show"]

        # load GT (cached)
        if dataset not in gt_cache:
            try:
                gt_cache[dataset] = find_gt_for_dataset(gt_dir, dataset)
            except RuntimeError as ex:
                print(f"Skipping dataset {dataset}: {ex}")
                continue
        gt = gt_cache[dataset]

        try:
            auc_raw = compute_auc(gt, pred)
            auc = normalize_auc_value(auc_raw)
        except Exception as ex:
            print(f"Error computing AUC for {fname}: {ex}")
            continue

        scores[dataset].append({
            "filename": fname,
            "args": args,
            "auc": auc
        })

    # print results per dataset
    for dataset, entries in scores.items():
        if not entries:
            continue
        print("\n" + "=" * 70)
        print(f"Dataset: {dataset}    (method: {METHOD_NAME})")
        entries_sorted = sorted(entries, key=lambda e: e["auc"], reverse=True)

        print("\nRanking (best → worst) — filename  |  AUC\n")
        for rank, e in enumerate(entries_sorted, start=1):
            print(f"{rank:2d}. {e['filename']}  |  AUC={e['auc']:.6f}")

        best = entries_sorted[0]
        print("\nBest (winner):")
        print(f"{best['filename']}  |  AUC={best['auc']:.6f}")
        print("=" * 70)


if __name__ == "__main__":
    main(RESULTS_DIR, GT_DIR)