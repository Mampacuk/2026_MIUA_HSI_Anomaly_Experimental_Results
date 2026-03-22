"""
kifd.py

Parallel, resumeable parameter search for KIFD with checkpoints and AUC-based selection.
"""

import os
import glob
import json
import shutil
import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import matlab.engine
import random

# ---------------- USER CONFIG ----------------
DATASET_DIR = "input130x174"                 # folder with .mat inputs
OUTPUT_DIR = "kifd_output"                   # outputs, checkpoints
MATLAB_SCRIPT_DIR = "matlab_source/KIFD"     # folder containing runKIFD.m
MAX_WORKERS = 11                             # adjust to your CPU / RAM
MAX_EVALS_PER_DATASET = 10000                # budget per dataset
PROGRESS_REPORT_EVERY = 10                   # print a progress line every eval
# --------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"{now()} | {msg}", flush=True)

def safe_auc(gt_mask, score_map):
    gt = gt_mask.ravel().astype(np.int32)
    sc = score_map.ravel().astype(float)
    if np.all(gt == 0) or np.all(gt == 1):
        return None
    try:
        return float(roc_auc_score(gt, sc))
    except Exception:
        return None

def load_ground_truth(mat_path):
    mat = sio.loadmat(mat_path)
    if 'map' in mat:
        gt = np.asarray(mat['map'])
    else:
        raise RuntimeError(f"{mat_path} lacks 'map'")
    return gt

# ---------------- Parameter grids ----------------
# We'll build coarse grids and allow per-dataset dynamic zeta (<= bands)
# - tree_num (q): default 1000 in paper; try 200..2000 coarse
# - subsample_percentage (M%): paper uses 3% × N; experiment around [1,2,3,5,8]
# - a_threshold "a": area threshold used in Local_iforest (absolute pixel count).
#   We'll parametrize as fractions of N: N/480, N/240, N/120, N/60

TREE_NUMS = [200, 500, 1000, 2000]
SUBSAMPLE_PCTS = [1, 2, 3, 5, 8]    # percents
A_FRACTIONS = [480, 240, 120, 60]    # denominator; a = max(1, round(N/denom))

# zeta list will be computed per-dataset (cap to bands). We'll choose:
# zeta candidates = {min(bands, 5), min(bands, 10), min(bands, 20), min(bands, bands)}
# but ensure unique and >=3
def compute_zeta_candidates(bands):
    c = sorted(set([max(3, min(bands, x)) for x in (5,10,20,bands)]))
    return c

# ---------------- Utility ----------------
def combo_key(zeta, tree_num, pct, a):
    return f"z{zeta}_q{tree_num}_m{pct}_a{a}"

def ensure_matlab_path(eng):
    eng.addpath(eng.genpath(MATLAB_SCRIPT_DIR), nargout=0)

def eval_combo_matlab(eng, mat_path, tmp_out, zeta, q, pct, a):
    # calls runKIFD(input_mat_path, output_mat_path, zeta, tree_num, subsample_percentage, a_threshold)
    eng.runKIFD(mat_path, tmp_out, float(zeta), int(q), float(pct), int(a), nargout=0)
    d = sio.loadmat(tmp_out)
    if 'show' not in d:
        return None
    return np.asarray(d['show'])

# ---------------- Worker for each dataset ----------------
def worker_process(mat_path):
    name = os.path.splitext(os.path.basename(mat_path))[0]
    log(f"[START] {name}")

    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"{name}_checkpoint.json")
    best_mat_path = os.path.join(OUTPUT_DIR, f"KIFD_best_{name}.mat")
    tmp_mat_path = os.path.join(OUTPUT_DIR, f"KIFD_temp_{name}.mat")

    gt = load_ground_truth(mat_path)
    gt_flat = gt.reshape(-1)
    H, W = gt.shape
    N = H * W

    # zeta candidates depends on bands: load bands
    mat = sio.loadmat(mat_path)
    if 'data' not in mat:
        raise RuntimeError(f"{name} has no 'data' variable")
    bands = mat['data'].shape[2]
    zeta_candidates = compute_zeta_candidates(bands)

    # build full coarse grid (zeta x tree_num x subsample_pct x a_options)
    a_options = [max(1, round(N / denom)) for denom in A_FRACTIONS]
    full_grid = [(z, q, m, a) for z in zeta_candidates for q in TREE_NUMS for m in SUBSAMPLE_PCTS for a in a_options]
    random.shuffle(full_grid)  # randomize order so partial runs cover space evenly
    total_combos = len(full_grid)

    # load checkpoint if exists
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            ckpt = json.load(f)
        tested = set(ckpt.get('tested', []))
        best_auc = ckpt.get('best_auc', None)
        best_params = tuple(ckpt.get('best_params')) if ckpt.get('best_params') else None
        records = ckpt.get('records', {})
        log(f"[RESUME] {name}: loaded checkpoint ({len(tested)} tested), best_auc={best_auc}")
    else:
        tested = set()
        best_auc = None
        best_params = None
        records = {}

    # start matlab engine
    eng = matlab.engine.start_matlab()
    ensure_matlab_path(eng)

    eval_count = 0
    for idx, (z, q, m, a) in enumerate(full_grid):
        key = combo_key(z, q, m, a)
        # stop if budget reached
        if eval_count >= MAX_EVALS_PER_DATASET:
            log(f"[BUDGET] {name}: reached budget {MAX_EVALS_PER_DATASET}")
            break
        if key in tested:
            continue

        try:
            show = eval_combo_matlab(eng, mat_path, tmp_mat_path, z, q, m, a)
        except Exception as e:
            log(f"[ERROR] {name} {key} MATLAB error: {e}")
            tested.add(key)
            # write checkpoint on error
            with open(ckpt_path, 'w') as f:
                json.dump({'tested': list(tested), 'records': records, 'best_auc': best_auc, 'best_params': list(best_params) if best_params else None, 'last_update': now()}, f)
            continue

        auc = safe_auc(gt_flat, show) if show is not None else None
        records[key] = auc
        tested.add(key)
        eval_count += 1

        # save temp mat (already written by MATLAB) and optionally copy to best
        if auc is not None and (best_auc is None or auc > best_auc):
            shutil.copyfile(tmp_mat_path, best_mat_path)
            best_auc = auc
            best_params = (z, q, m, a)
            log(f"[IMPROVE] {name} NEW BEST auc={best_auc:.6f} params={best_params}")

        # periodic progress report
        if (eval_count % PROGRESS_REPORT_EVERY) == 0:
            tried = len(tested)
            percent = tried / total_combos * 100.0
            log(f"[PROGRESS] {name}: eval {eval_count}/{MAX_EVALS_PER_DATASET}, tried {tried}/{total_combos} ({percent:.1f}%) remaining {total_combos - tried}")

        # checkpoint persist
        ckpt = {'tested': list(tested), 'records': records, 'best_auc': best_auc, 'best_params': list(best_params) if best_params else None, 'last_update': now()}
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)

    eng.quit()
    log(f"[DONE] {name}: best_auc={best_auc} best_params={best_params} evals={eval_count}")
    return {'dataset': name, 'best_auc': best_auc, 'best_params': best_params, 'evaluations': eval_count, 'checkpoint': ckpt_path, 'best_mat': best_mat_path if os.path.exists(best_mat_path) else None}

def main():
    mat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mat")))
    if not mat_files:
        raise RuntimeError("No .mat files found in DATASET_DIR")
    log(f"Found {len(mat_files)} datasets. Starting with up to {MAX_WORKERS} workers.")

    results = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(mat_files))) as ex:
        for r in ex.map(worker_process, mat_files):
            results.append(r)
            log(f"[FINISHED] {r['dataset']} -> best_auc={r['best_auc']} best_params={r['best_params']}")

    log("All done. Summary:")
    for r in results:
        log(f"  {r['dataset']}: best_auc={r['best_auc']} best_params={r['best_params']} best_mat={r['best_mat']} checkpoint={r['checkpoint']}")

if __name__ == '__main__':
    main()