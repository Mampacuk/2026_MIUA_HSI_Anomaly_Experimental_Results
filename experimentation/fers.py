"""
Hyperparameter grid search runner for your 3-stage hyperspectral anomaly detection pipeline.
- Edit the constants below to point to your train1.py, train2.py, test.py and the data folders.
- Run this script on the same machine where you run those scripts (they will be called via subprocess).
- The script keeps a JSON checkpoint so you can stop & resume safely.
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
import re

# Optional but recommended: these libraries are used for sanity cross-check from .mat files
try:
    import scipy.io as scio
    import numpy as np
    from sklearn.metrics import roc_auc_score
except Exception as e:
    scio = None
    np = None
    roc_auc_score = None
    # We'll still operate, but mat-based cross-check won't run.

# ---------------------- USER EDIT THIS SECTION ----------------------
# Absolute paths to the scripts (hardcode these as you requested)
TRAIN1_PATH = r"D:\dev\FERS\train_FERD\train1.py"
TRAIN2_PATH = r"D:\dev\FERS\train2.py"
TEST_PATH   = r"D:\dev\FERS\FERS_ABU\test.py"

# The background training data used by train1/train2 (adjust as needed)
# BACKGROUND_TRAIN_PATH = r"D:\dev\experimental_results\experimentation\input\input260x348_bg_crops64x64_stride48_porcine"   # usually background crops folder (train)
# BACKGROUND_TEST_PATH  = r"D:\dev\experimental_results\experimentation\input\input130x174_porcine"    # used by train1/train internal test during training
BACKGROUND_TRAIN_PATH = r"D:\dev\experimental_results\experimentation\input\input260x348_bg_crops64x64_stride64_bovine"   # usually background crops folder (train)
BACKGROUND_TEST_PATH  = r"D:\dev\experimental_results\experimentation\input\input128x174_bovine"    # used by train1/train internal test during training

# The dataset you want to evaluate (HSIs .mat files with ground truth "map" variable)
EVAL_DATA_PATH = BACKGROUND_TEST_PATH        # the dataset the user described (test images + ground truth "map")

# Where to store sweep outputs and checkpoint
# RESULTS_ROOT = r"D:\dev\experimental_results\experimentation\output\fers_output\porcine"
RESULTS_ROOT = r"D:\dev\experimental_results\experimentation\output\fers_output\bovine"
CHECKPOINT_JSON = os.path.join(RESULTS_ROOT, "sweep_checkpoint.json")
BEST_MAT_DIR = os.path.join(RESULTS_ROOT, "best_result_mats")

# Python executable used to call scripts (set to sys.executable by default)
PYTHON_BIN = r"D:\dev\IEEE_JSTARS_NL2Net\.venv\Scripts\python.exe"
# ------------------------------------------------------------------

# Recommended hyperparameter grid (sensible defaults for lesions larger than remote-sensing tiny anomalies)
# I keep the grid small but useful — expand values if you want a broader sweep.
HYPERPARAM_GRID = {
    # Stage1 hyperparams (train1.py)
    "seed": [1, 10, 42],
    "input_channel": [31],          # keep as your sensor bands; change if necessary
    "lr_stage1": [0.005, 0.001],
    "batch_stage1": [8, 16],
    "epochs_stage1": [40, 60],
    "mse_loss_stage1": [0.1, 1.0],
    "cos_loss_stage1": [0.1, 1.0],

    # Stage2 hyperparams (train2.py)
    "lr_stage2": [0.005, 0.001],
    "batch_stage2": [12, 18],
    "epochs_stage2": [40, 60],
    "mse_loss_stage2": [1.0],        # This is the 'mse_loss' arg in train2.py
    "ssim_loss_stage2": [0.01, 0.05],

    # Test-time choices
    "detect": ["AE", "RX"],

    # Other practical knobs
    # Limit how many combinations you test by adjusting these lists
}

# ---------------------- utility functions ----------------------
def ensure_dirs():
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(BEST_MAT_DIR).mkdir(parents=True, exist_ok=True)

def param_dict_to_key(p):
    # create a stable short key for the parameter dict (deterministic)
    s = json.dumps(p, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def generate_grid(grid):
    # produce dicts for each combination
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for comb in itertools.product(*values):
        yield dict(zip(keys, comb))

def safe_dump_checkpoint(chk):
    tmp = CHECKPOINT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(chk, f, indent=2)
    os.replace(tmp, CHECKPOINT_JSON)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_JSON):
        with open(CHECKPOINT_JSON, "r") as f:
            return json.load(f)
    else:
        return {
            "best_mean_auc": -1.0,
            "best_params": None,
            "tried": [],
            "results": []
        }

def read_mean_auc_from_log(log_path):
    """Parse log.txt for 'mean pixel ROCAUC:' last occurrence."""
    if not os.path.exists(log_path):
        return None
    val = None
    with open(log_path, "r") as f:
        for line in f:
            if "mean pixel ROCAUC" in line:
                # find float
                m = re.search(r"mean pixel ROCAUC[: ]+([0-9]*\.[0-9]+)", line)
                if m:
                    val = float(m.group(1))
    return val

def parse_any_floats_in_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        text = f.read()
    floats = re.findall(r"[0-9]+\.[0-9]+", text)
    return [float(x) for x in floats]

def cross_check_each_auc(each_auc_path):
    # read all floats in each_auc.txt and compute their mean (if reasonable)
    floats = parse_any_floats_in_file(each_auc_path)
    if floats:
        # heuristic: if more than one float, average them
        return sum(floats)/len(floats)
    return None

def compute_mean_auc_from_mats(scores_dir, eval_data_path):
    """
    Try to compute mean AUC using saved score .mat files vs their ground truth in eval_data_path.
    Assumes filenames match (basename) and the ground truth file contains variable 'map' (as you described).
    Returns mean AUC if possible, otherwise None.
    """
    if scio is None or np is None or roc_auc_score is None:
        return None

    if not os.path.isdir(scores_dir):
        return None

    score_files = sorted([p for p in Path(scores_dir).glob("*.mat")])
    if not score_files:
        return None

    aucs = []
    for sf in score_files:
        basename = sf.stem
        # result mat stored variable 'show' according to your test.py
        try:
            sdat = scio.loadmat(str(sf))
            if "show" not in sdat:
                continue
            score_map = np.squeeze(sdat["show"]).astype(float).flatten()
        except Exception:
            continue

        # attempt to find corresponding eval mat file (same basename) in eval_data_path
        candidate = Path(eval_data_path) / (basename + ".mat")
        if not candidate.exists():
            # try search by wildcard
            found = list(Path(eval_data_path).glob(basename + "*"))
            candidate = found[0] if found else None
        if not candidate or not candidate.exists():
            continue
        try:
            m = scio.loadmat(str(candidate))
            if "map" in m:
                gt = np.squeeze(m["map"]).flatten()
            else:
                continue
            # ensure binary 0/1
            gtbin = (gt == 1)
            # compute roc_auc
            a = roc_auc_score(gtbin, score_map)
            aucs.append(a)
        except Exception:
            continue

    if aucs:
        return float(np.mean(aucs))
    return None

# ---------------------- runner core ----------------------
def run_command(cmd, cwd=None, logfile=None):
    """Run a command (list) and stream to logfile (path) and to stdout."""
    print("RUN:", " ".join(cmd))
    os.makedirs(os.path.dirname(logfile), exist_ok=True) if logfile else None
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        out_lines = []
        with open(logfile, "a") as lf:
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                out_lines.append(line)
                print(line, end="")
        ret = p.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return out_lines

def run_single_iteration(params, iteration_dir):
    """
    Executes:
      - stage1: train1.py with params
      - stage2: train2.py with params (checkpoint_dir -> stage1 saved models)
      - test:    test.py with params (checkpoint_dir -> stage1 saved models, save_dir -> iteration_dir)
    Returns dict with mean_auc and debug paths.
    """
    # ensure iteration dir
    os.makedirs(iteration_dir, exist_ok=True)

    # get working directories for scripts
    train1_cwd = os.path.dirname(os.path.abspath(TRAIN1_PATH)) or "."
    train2_cwd = os.path.dirname(os.path.abspath(TRAIN2_PATH)) or "."
    test_cwd   = os.path.dirname(os.path.abspath(TEST_PATH)) or "."

    # stage1 seed and saved dir conventions (train1.py saves models to ./result/seed{seed}/saved_models)
    seed = params["seed"]
    stage1_seed_label = f"seed{seed}"
    # train1 saves to ./result/seed{seed}
    stage1_result_base = os.path.join(train1_cwd, "result", stage1_seed_label)
    stage1_saved_models = os.path.join(stage1_result_base, "saved_models")

    # Build train1 command
    cmd1 = [
        PYTHON_BIN, TRAIN1_PATH,
        "--train_data_path", BACKGROUND_TRAIN_PATH,
        "--test_data_path", BACKGROUND_TEST_PATH,
        "--input_channel", str(params.get("input_channel", 31)),
        "--epochs", str(params.get("epochs_stage1", 60)),
        "--batch_size", str(params.get("batch_stage1", 16)),
        "--lr", str(params.get("lr_stage1", 0.005)),
        "--seed", str(seed),
        "--mse_loss", str(params.get("mse_loss_stage1", 0.1)),
        "--cos_loss", str(params.get("cos_loss_stage1", 1.0)),
        "--save_txt", "True",
        "--save_model", "True"
    ]
    # logfile for stage1
    log1 = os.path.join(iteration_dir, "train1.log")
    run_command(cmd1, cwd=train1_cwd, logfile=log1)

    # Verify stage1 models exist
    # convh.pt etc should be in stage1_saved_models
    if not os.path.isdir(stage1_saved_models):
        # maybe saved in relative path to script, so check alternative path
        raise FileNotFoundError(f"Stage1 saved_models dir not found at {stage1_saved_models}")

    # stage2: train2.py; ensure checkpoint_dir -> stage1_saved_models
    cmd2 = [
        PYTHON_BIN, TRAIN2_PATH,
        "--train_data_path", BACKGROUND_TRAIN_PATH,
        "--test_data_path", BACKGROUND_TEST_PATH,
        "--input_channel", str(params.get("input_channel", 31)),
        "--epochs", str(params.get("epochs_stage2", 60)),
        "--batch_size", str(params.get("batch_stage2", 18)),
        "--lr", str(params.get("lr_stage2", 0.005)),
        "--seed", str(seed),
        "--checkpoint_dir", stage1_saved_models,
        "--mse_loss", str(params.get("mse_loss_stage2", 1.0)),
        "--ssim_loss", str(params.get("ssim_loss_stage2", 0.01)),
        "--save_txt", "True",
        "--save_model", "True"
    ]
    log2 = os.path.join(iteration_dir, "train2.log")
    run_command(cmd2, cwd=train2_cwd, logfile=log2)

    # After stage2, spa_fen.pt should exist in stage1_saved_models (train2 writes it to args.saved_models_dir which is set to checkpoint_dir)
    # stage2 saved models dir:
    stage2_saved_models = stage1_saved_models
    if not os.path.isdir(stage2_saved_models):
        raise FileNotFoundError(f"Stage2 saved_models dir not found at {stage2_saved_models}")

    # stage3 / test
    # give test a unique save_dir per iteration so its outputs go to iteration_dir/final_result
    test_save_dir = os.path.join(iteration_dir, "final_result")
    os.makedirs(test_save_dir, exist_ok=True)

    cmd3 = [
        PYTHON_BIN, TEST_PATH,
        "--data_path", EVAL_DATA_PATH,
        "--input_channel", str(params.get("input_channel", 31)),
        "--seed", str(seed),
        "--detect", params.get("detect", "AE"),
        "--checkpoint_dir", stage2_saved_models,
        "--save_dir", test_save_dir
    ]
    log3 = os.path.join(iteration_dir, "test.log")
    run_command(cmd3, cwd=test_cwd, logfile=log3)

    # parse results
    # test.py writes log at <test_save_dir>/log.txt and each_auc.txt, and saves mats in <test_save_dir>/scores
    mean_from_log = read_mean_auc_from_log(os.path.join(test_save_dir, "log.txt"))
    mean_from_each = cross_check_each_auc(os.path.join(test_save_dir, "each_auc.txt"))

    mean_from_mat = None
    try:
        mean_from_mat = compute_mean_auc_from_mats(os.path.join(test_save_dir, "scores"), EVAL_DATA_PATH)
    except Exception:
        mean_from_mat = None

    # decide final mean to use: prefer log, then each_auc, then mats
    mean_auc = mean_from_log if mean_from_log is not None else (mean_from_each if mean_from_each is not None else mean_from_mat)

    return {
        "mean_auc": mean_auc,
        "mean_from_log": mean_from_log,
        "mean_from_each": mean_from_each,
        "mean_from_mat": mean_from_mat,
        "iteration_dir": iteration_dir,
        "test_save_dir": test_save_dir,
        "stage1_saved_models": stage1_saved_models
    }


# ---------------------- main sweep loop ----------------------
def main():
    ensure_dirs()
    checkpoint = load_checkpoint()

    # build grid generator
    gridgen = list(generate_grid(HYPERPARAM_GRID))
    print(f"Total combinations in grid: {len(gridgen)}")
    print("FIRST FEW param combos (for inspection):")
    for x in gridgen[:6]:
        print("  ", x)

    try:
        for params in gridgen:
            # skip if tried
            if params in checkpoint.get("tried", []):
                print("Skipping already tried combo:", params)
                continue

            key = param_dict_to_key(params)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            iteration_dir = os.path.join(RESULTS_ROOT, f"{ts}_{key}")
            print(f"\n=== Starting iteration {ts} key={key} ===\nparams: {json.dumps(params)}\niteration_dir: {iteration_dir}\n")

            try:
                res = run_single_iteration(params, iteration_dir)
            except Exception as e:
                print(f"ERROR during iteration {key}: {e}")
                # mark as tried (so we don't repeat the exact failing combination),
                # but record the error so you can inspect the logs in iteration_dir.
                checkpoint.setdefault("tried", []).append(params)
                checkpoint.setdefault("results", []).append({
                    "params": params,
                    "error": str(e),
                    "iteration_dir": iteration_dir,
                    "time": datetime.utcnow().isoformat()
                })
                safe_dump_checkpoint(checkpoint)
                # continue with next combo
                continue

            mean_auc = res.get("mean_auc")
            print(f"Iteration result: mean_auc={mean_auc} (log: {res.get('mean_from_log')}, each_auc: {res.get('mean_from_each')}, mat_check: {res.get('mean_from_mat')})")

            # Copy mats to a safe place under iteration dir if present (already in test_save_dir/scores)
            scores_src = os.path.join(res["test_save_dir"], "scores")
            if os.path.isdir(scores_src):
                dest_scores = os.path.join(iteration_dir, "saved_scores")
                if os.path.exists(dest_scores):
                    shutil.rmtree(dest_scores)
                shutil.copytree(scores_src, dest_scores, dirs_exist_ok=True)

            # update checkpoint
            checkpoint.setdefault("tried", []).append(params)
            checkpoint.setdefault("results", []).append({
                "params": params,
                "mean_auc": mean_auc,
                "iteration_dir": iteration_dir,
                "test_save_dir": res["test_save_dir"],
                "time": datetime.utcnow().isoformat()
            })

            # check if better than best
            if mean_auc is not None and mean_auc > checkpoint.get("best_mean_auc", -1.0):
                print(f"NEW BEST mean_auc {mean_auc} > previous {checkpoint.get('best_mean_auc')}")
                checkpoint["best_mean_auc"] = float(mean_auc)
                checkpoint["best_params"] = params
                # copy all .mat files from test_save_dir/scores to BEST_MAT_DIR (overwrite)
                best_scores = os.path.join(res["test_save_dir"], "scores")
                if os.path.isdir(best_scores):
                    # clear BEST_MAT_DIR before copying
                    for f in os.listdir(BEST_MAT_DIR):
                        try:
                            os.remove(os.path.join(BEST_MAT_DIR, f))
                        except Exception:
                            pass
                    for mf in Path(best_scores).glob("*.mat"):
                        try:
                            shutil.copy2(str(mf), BEST_MAT_DIR)
                        except Exception:
                            pass
                # also copy iteration logs for convenience
                try:
                    shutil.copytree(iteration_dir, os.path.join(BEST_MAT_DIR, "best_iteration"), dirs_exist_ok=True)
                except Exception:
                    pass

            safe_dump_checkpoint(checkpoint)

        print("Grid search finished.")
        print("Best so far:", checkpoint.get("best_mean_auc"), checkpoint.get("best_params"))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint and exiting.")
        safe_dump_checkpoint(checkpoint)
        sys.exit(0)


if __name__ == "__main__":
    main()