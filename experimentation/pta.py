#!/usr/bin/env python3
"""
ptasearch_parallel.py

Parallel adaptive parameter search for PTA via matlab.engine.

Behavior:
- One process per dataset (ProcessPoolExecutor).
- Each worker:
    - loads ground-truth 'map' from the input .mat,
    - does a coarse grid search (evaluate AUC for each combo),
    - selects top-k coarse combos and performs local refinements (finer mu/beta around best),
    - for each combo: calls MATLAB runPTA(...) which writes a TEMP .mat,
      Python immediately reads TEMP, computes AUC, updates BEST if improved,
    - writes checkpoint JSON to allow resume.
"""

import os
import glob
import json
import shutil
import itertools
import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score

import matlab.engine

# -------------------------
# USER CONFIG
# -------------------------
DATASET_DIR = "input130x174"               # folder with .mat inputs
OUTPUT_DIR = "pta_output"                  # results & best files
MATLAB_SCRIPT_DIR = "matlab_source/PTA"    # folder containing runPTA.m
MAX_WORKERS = 11                           # number of parallel dataset processes (adjust)
COARSE_TOPK = 6                            # number of top coarse combos to refine
MAX_EVALS_PER_DATASET = 100000             # budget per dataset (coarse+refinements)
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"{now()} | {msg}", flush=True)

# -------------------------
# PARAMETER RANGES (coarse -> refine)
# RATIONALE:
# - PTA paper used r ∈ [0..20], hyperparams α, μ, β, τ across powers of 10.
# - μ and β found to be the most sensitive in the paper; we will refine those.
# - For biomedical data (higher anomaly proportion, larger contiguous lesions,
#   fewer bands, UV low-SNR), we bias towards:
#     * smaller r (background spectral rank tends to be low with few bands),
#     * exploration of μ over a range that allows stronger penalty (larger μ),
#     * β smaller values (group sparsity less severe since anomalies occupy more pixels).
# -------------------------

# Coarse ranges
r_coarse     = [0, 2, 5, 10]                      # truncate_rank (paper: <20; our data -> smaller)
alpha_coarse = [1e-2, 1e-1, 1.0, 10.0]            # α (low-rank weight)
mu_coarse    = [1e-3, 1e-2, 1e-1, 1.0]            # μ (penalty) - explore wide
beta_coarse  = [1e-4, 1e-3, 1e-2, 1e-1]           # β (group-sparse) - lower favored
tau_coarse   = [0.1, 1.0, 10.0]                   # τ
maxiter = 400

# Refinement multipliers for mu & beta (geometric)
refine_factors = [0.25, 0.5, 1.0, 2.0, 4.0]

# global budget per dataset (coarse evals + refinements)
# -------------------------

def combo_key(params):
    r, a, mu, b, tau = params
    return f"r{int(r)}_a{a:g}_mu{mu:g}_b{b:g}_tau{tau:g}"

def safe_auc(gt_mask, score_map):
    """
    Compute AUC (roc_auc_score) with safety checks.
    gt_mask: flattened binary labels (0/1)
    score_map: flattened scores (float)
    """
    gt = gt_mask.ravel().astype(np.int32)
    sc = score_map.ravel().astype(float)
    # need both classes present
    if np.all(gt == 0) or np.all(gt == 1):
        return None
    try:
        return float(roc_auc_score(gt, sc))
    except Exception as e:
        return None

def load_ground_truth(mat_path):
    mat = sio.loadmat(mat_path)
    if 'map' in mat:
        gt = np.asarray(mat['map'])
    else:
        raise RuntimeError(f"Input {mat_path} lacks 'map'")
    return gt

def ensure_matlab_engine_path(eng, matlab_script_dir):
    eng.addpath(eng.genpath(matlab_script_dir), nargout=0)

def evaluate_combo_with_matlab(eng, mat_path, tmp_out_path, params):
    """
    Call MATLAB runPTA to write tmp_out_path; return AUC (or None).
    params: (r, alpha, mu, beta, tau)
    """
    r, alpha, mu, beta, tau = params
    # call runPTA(input, output, truncate_rank, alphia, mu, beta, tau, maxiter)
    eng.runPTA(mat_path, tmp_out_path, float(r), float(alpha), float(mu), float(beta), float(tau), int(maxiter), nargout=0)
    data = sio.loadmat(tmp_out_path)
    if 'show' not in data:
        log(f"[WARN] MATLAB output {tmp_out_path} missing 'show'")
        return None, None
    show = np.asarray(data['show'])
    return show, tmp_out_path

def worker_process(mat_path):
    """
    Worker that performs adaptive parameter search for one dataset.
    This function is invoked in a separate process (ProcessPoolExecutor).
    """
    name = os.path.splitext(os.path.basename(mat_path))[0]
    log(f"[START] {name}")

    # checkpoint file per dataset
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"{name}_checkpoint.json")
    best_mat_path = os.path.join(OUTPUT_DIR, f"PTA_best_{name}.mat")
    temp_mat_path = os.path.join(OUTPUT_DIR, f"PTA_temp_{name}.mat")

    # load ground truth (done in Python to compute AUC)
    gt = load_ground_truth(mat_path)
    gt_flat = gt.reshape(-1)

    # load checkpoint if present
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            ckpt = json.load(f)
        tested = set(ckpt.get('tested', []))
        best_auc = ckpt.get('best_auc', None)
        best_params = tuple(ckpt.get('best_params')) if ckpt.get('best_params') else None
        log(f"[RESUME] {name} - loaded checkpoint: {len(tested)} tested, best_auc={best_auc}")
    else:
        tested = set()
        best_auc = None
        best_params = None
        ckpt = {}

    # start MATLAB engine (one per worker process)
    eng = matlab.engine.start_matlab()
    ensure_matlab_engine_path(eng, MATLAB_SCRIPT_DIR)

    # --- STEP 1: coarse grid ---
    coarse_combos = list(itertools.product(r_coarse, alpha_coarse, mu_coarse, beta_coarse, tau_coarse))
    log(f"[COARSE] {name} will evaluate {len(coarse_combos)} coarse combos")

    evals = 0
    records = {}  # mapping combo_key -> auc

    # If checkpoint had 'records' we can reuse to skip evals
    if 'records' in ckpt:
        for k, v in ckpt['records'].items():
            records[k] = v

    for params in coarse_combos:
        key = combo_key(params)
        if key in tested:
            continue
        if evals >= MAX_EVALS_PER_DATASET:
            log(f"[BUDGET] {name} reached MAX_EVALS_PER_DATASET during coarse")
            break

        try:
            show, tmp_path = evaluate_combo_with_matlab(eng, mat_path, temp_mat_path, params)
        except Exception as e:
            log(f"[ERROR] {name} coarse params {key} matlab error: {e}")
            tested.add(key)
            continue

        auc = safe_auc(gt_flat, show)
        records[key] = auc
        tested.add(key)
        evals += 1
        log(f"[EVAL] {name} coarse {key} -> auc={auc}")

        # maintain best
        if auc is not None and (best_auc is None or auc > best_auc):
            shutil.copyfile(temp_mat_path, best_mat_path)
            best_auc = auc
            best_params = params
            log(f"[IMPROVE] {name} NEW BEST auc={best_auc} params={best_params}")

        # checkpoint update
        ckpt.update({
            'tested': list(tested),
            'records': records,
            'best_auc': best_auc,
            'best_params': list(best_params) if best_params else None,
            'last_update': now()
        })
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)

    # if no successful coarse evals, exit
    if len(records) == 0:
        log(f"[FAIL] {name} no coarse results; quitting")
        eng.quit()
        return

    # pick top-K coarse combos to refine (based on auc; treat None as -inf)
    scored = [(k, v) for k, v in records.items()]
    scored_sorted = sorted(scored, key=lambda kv: (kv[1] is not None, kv[1]), reverse=True)
    topk = [k for k, v in scored_sorted[:COARSE_TOPK] if v is not None]

    # convert key back to params helper
    def key_to_params(kstr):
        # key format: r{r}_a{a}_mu{mu}_b{b}_tau{tau}
        parts = kstr.split('_')
        r = int(parts[0][1:])
        a = float(parts[1][1:])
        mu = float(parts[2][2:])
        b = float(parts[3][1:])
        tau = float(parts[4][3:])
        return (r, a, mu, b, tau)

    # --- STEP 2: refinements around best coarse combos ---
    log(f"[REFINE] {name} refining top {len(topk)} coarse combos")

    for kstr in topk:
        base = key_to_params(kstr)
        r0, a0, mu0, b0, tau0 = base

        # generate refined mu & beta values around mu0, b0 using refine_factors
        mu_vals = sorted(set(max(1e-12, mu0 * f) for f in refine_factors))
        b_vals = sorted(set(max(1e-12, b0 * f) for f in refine_factors))

        # keep same r,a,tau but try refined mu,beta pairs
        for mu_val in mu_vals:
            for b_val in b_vals:
                params = (r0, a0, mu_val, b_val, tau0)
                key = combo_key(params)
                if key in tested:
                    continue
                if evals >= MAX_EVALS_PER_DATASET:
                    log(f"[BUDGET] {name} reached MAX_EVALS_PER_DATASET during refine")
                    break

                try:
                    show, tmp_path = evaluate_combo_with_matlab(eng, mat_path, temp_mat_path, params)
                except Exception as e:
                    log(f"[ERROR] {name} refine params {key} matlab error: {e}")
                    tested.add(key)
                    continue

                auc = safe_auc(gt_flat, show)
                records[key] = auc
                tested.add(key)
                evals += 1
                log(f"[EVAL] {name} refine {key} -> auc={auc}")

                if auc is not None and (best_auc is None or auc > best_auc):
                    shutil.copyfile(temp_mat_path, best_mat_path)
                    best_auc = auc
                    best_params = params
                    log(f"[IMPROVE] {name} NEW BEST auc={best_auc} params={best_params}")

                # checkpoint update
                ckpt.update({
                    'tested': list(tested),
                    'records': records,
                    'best_auc': best_auc,
                    'best_params': list(best_params) if best_params else None,
                    'last_update': now()
                })
                with open(ckpt_path, 'w') as f:
                    json.dump(ckpt, f)

            if evals >= MAX_EVALS_PER_DATASET:
                break
        if evals >= MAX_EVALS_PER_DATASET:
            break

    log(f"[DONE] {name} best_auc={best_auc} best_params={best_params} total_evals={evals}")
    eng.quit()
    return {
        'dataset': name,
        'best_auc': best_auc,
        'best_params': best_params,
        'evaluations': evals,
        'checkpoint': ckpt_path,
        'best_mat': best_mat_path if os.path.exists(best_mat_path) else None
    }

def main():
    mat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mat")))
    if not mat_files:
        raise RuntimeError("No .mat files found in DATASET_DIR")

    log(f"Found {len(mat_files)} datasets. Starting parallel search with up to {MAX_WORKERS} workers.")

    results = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(mat_files))) as ex:
        for res in ex.map(worker_process, mat_files):
            results.append(res)
            log(f"[FINISHED] {res['dataset']} -> best_auc={res['best_auc']} best_params={res['best_params']}")

    # summary
    log("All dataset searches finished. Summary:")
    for r in results:
        log(f"  {r['dataset']}: best_auc={r['best_auc']} best_params={r['best_params']} best_mat={r['best_mat']} checkpoint={r['checkpoint']}")

if __name__ == "__main__":
    main()