#!/usr/bin/env python3
"""
Priority-ordered hyperparameter search runner for HADGAN.

What it does
------------
- Scans all `.mat` datasets inside the porcine and bovine roots.
- Runs leave-one-out cross-validation within each group:
    train on each dataset, infer on every other dataset in the same group.
- Sweeps a prioritized hyperparameter queue (best-guess configs first).
- Stores every trial artifact:
    - checkpoint directory
    - inference `.mat`
    - per-pair JSON metadata
    - per-trial JSON summary
    - global resumable state JSON
- Maintains best results per ordered train/test pair and best overall mean-AUC.

Notes
-----
- Edit the top-level constants below before running.
- This runner assumes `train.py` and `infer.py` can be called from the shell.
- The search queue is intentionally priority-ordered instead of a blind full cartesian
  explosion, because full exhaustive CV over all combinations becomes enormous.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import scipy.io as sio


# =========================
# EDIT THESE 3 CONSTANTS
# =========================
PYTHON_EXE = r"C:\Users\alexa\miniconda3\envs\HADGAN\python.exe"
PORCINE_DIR = r"D:\dev\experimental_results\experimentation\input\input130x174_porcine"
BOVINE_DIR = r"D:\dev\experimental_results\experimentation\input\input128x174_bovine"

# Optional defaults; can be overridden on CLI
TRAIN_SCRIPT = r"D:\dev\HADGAN\HADGAN\train_hadgan.py"
INFER_SCRIPT = r"D:\dev\HADGAN\HADGAN\infer_hadgan.py"

# Root folder that will hold all search artifacts and the resumable state
WORK_DIR = Path(r"D:\dev\experimental_results\experimentation\output\hadgan_output")

# Primary metric for ranking trials and per-pair winners
PRIMARY_METRIC = "roc_auc"   # infer.py stores both roc_auc and pr_auc
SECONDARY_METRIC = "pr_auc"

# Retry policy for transient subprocess issues
MAX_RETRIES = 2

# If True, failed pair evaluations are retried on the next launch
RETRY_FAILED_PAIRS_ON_RESUME = True


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, file_handle=None) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if file_handle is not None:
        file_handle.write(line + "\n")
        file_handle.flush()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(path)


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_config(cfg: Dict[str, Any]) -> str:
    return hashlib.sha1(canonical_json(cfg).encode("utf-8")).hexdigest()[:12]


def normalize_dataset_name(path: Path) -> str:
    return path.stem


def find_mat_datasets(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    mats = []
    for p in root_path.rglob("*.mat"):
        # Keep raw datasets; ignore obvious derived outputs if the user stores them in the same tree.
        lower = p.name.lower()
        if any(tok in lower for tok in ("result", "trained_with", "checkpoint", "ckpt", "infer", "pred")):
            continue
        mats.append(p)
    mats = sorted({p.resolve() for p in mats}, key=lambda x: x.name.lower())
    return list(mats)


def paired_tasks(group: str, datasets: List[Path]) -> List[Tuple[str, str, Path, Path]]:
    tasks = []
    for train_path in datasets:
        for test_path in datasets:
            if train_path == test_path:
                continue
            tasks.append((group, normalize_dataset_name(train_path), train_path,
                          normalize_dataset_name(test_path), test_path))
    return tasks


@dataclass(frozen=True)
class TrialConfig:
    patch_size: int
    overlap: int
    dropout: float
    k_steps: int
    lr_encoder: float
    lr_others: float
    alpha0: float
    alpha1: float
    alpha2: float
    epochs: int
    method: str
    lbd: float
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ANCHOR = TrialConfig(
    patch_size=64,
    overlap=50,
    dropout=0.5,
    k_steps=4,
    lr_encoder=5e-5,
    lr_others=1e-4,
    alpha0=5.0,
    alpha1=1.0,
    alpha2=0.1,
    epochs=300,
    method="mean",
    lbd=0.5,
    seed=42,
)

# Ordered candidate values (best-guess first, less likely later)
CANDIDATES = {
    "patch_size": [64, 32, 120, 16],
    "overlap": [50, 75, 25, 0],
    "dropout": [0.5, 0.35, 0.65, 0.2, 0.0, 0.8],
    "k_steps": [4, 2, 6, 1, 8],
    "lr_encoder": [5e-5, 1e-4, 2e-5, 2e-4],
    "lr_others": [1e-4, 5e-5, 2e-4, 2e-5],
    "alpha0": [5.0, 2.0, 8.0, 1.0, 10.0],
    "alpha1": [1.0, 2.0, 0.5, 4.0],
    "alpha2": [0.1, 0.05, 0.2, 0.01, 0.5],
    "epochs": [300, 450, 200, 600],
    "method": ["mean", "weighted"],
    "lbd": [0.5, 0.65, 0.35, 0.8, 0.2],
}

# A compact, priority-ordered queue:
# 1) anchor
# 2) single-parameter moves
# 3) a few high-impact pairwise interactions
# 4) several "tail" edge cases
def build_priority_queue(max_trials: Optional[int] = None) -> List[TrialConfig]:
    queue: List[TrialConfig] = []
    seen = set()

    def add(cfg: TrialConfig):
        key = hash_config(cfg.to_dict())
        if key not in seen:
            seen.add(key)
            queue.append(cfg)

    # 1) Anchor first
    add(ANCHOR)

    # 2) Single-parameter moves from anchor, ordered by closeness
    for field in ["patch_size", "overlap", "dropout", "k_steps", "lr_encoder", "lr_others", "alpha0", "alpha1", "alpha2", "epochs", "method", "lbd"]:
        for v in CANDIDATES[field]:
            if v == getattr(ANCHOR, field):
                continue
            cfg = ANCHOR.__dict__.copy()
            cfg[field] = v
            add(TrialConfig(**cfg))

    # 3) High-impact pairwise interactions
    # patch_size x overlap
    for patch_size in CANDIDATES["patch_size"]:
        for overlap in CANDIDATES["overlap"]:
            cfg = ANCHOR.__dict__.copy()
            cfg["patch_size"] = patch_size
            cfg["overlap"] = overlap
            add(TrialConfig(**cfg))

    # patch_size x dropout
    for patch_size in [64, 32, 120]:
        for dropout in [0.5, 0.35, 0.65, 0.2]:
            cfg = ANCHOR.__dict__.copy()
            cfg["patch_size"] = patch_size
            cfg["dropout"] = dropout
            add(TrialConfig(**cfg))

    # k_steps x dropout
    for k_steps in [4, 2, 6]:
        for dropout in [0.5, 0.35, 0.65]:
            cfg = ANCHOR.__dict__.copy()
            cfg["k_steps"] = k_steps
            cfg["dropout"] = dropout
            add(TrialConfig(**cfg))

    # learning-rate pair
    for lr_enc in [5e-5, 1e-4, 2e-5]:
        for lr_oth in [1e-4, 5e-5, 2e-4]:
            cfg = ANCHOR.__dict__.copy()
            cfg["lr_encoder"] = lr_enc
            cfg["lr_others"] = lr_oth
            add(TrialConfig(**cfg))

    # alpha triplet focused neighborhood
    for alpha0 in [5.0, 2.0, 8.0]:
        for alpha1 in [1.0, 2.0]:
            for alpha2 in [0.1, 0.05, 0.2]:
                cfg = ANCHOR.__dict__.copy()
                cfg["alpha0"] = alpha0
                cfg["alpha1"] = alpha1
                cfg["alpha2"] = alpha2
                add(TrialConfig(**cfg))

    # method / lbd
    for method in ["mean", "weighted"]:
        for lbd in [0.5, 0.65, 0.35, 0.8]:
            cfg = ANCHOR.__dict__.copy()
            cfg["method"] = method
            cfg["lbd"] = lbd
            add(TrialConfig(**cfg))

    # epochs / overlap / patch-size tail
    for epochs in [300, 450, 200]:
        for patch_size in [64, 32]:
            for overlap in [50, 75]:
                cfg = ANCHOR.__dict__.copy()
                cfg["epochs"] = epochs
                cfg["patch_size"] = patch_size
                cfg["overlap"] = overlap
                add(TrialConfig(**cfg))

    # 4) Edge cases / unlikely tail
    tail_configs = [
        dict(patch_size=16, overlap=0, dropout=0.0, k_steps=1, lr_encoder=2e-4, lr_others=2e-5,
             alpha0=1.0, alpha1=4.0, alpha2=0.5, epochs=200, method="weighted", lbd=0.2),
        dict(patch_size=16, overlap=75, dropout=0.8, k_steps=8, lr_encoder=2e-5, lr_others=2e-4,
             alpha0=10.0, alpha1=0.5, alpha2=0.01, epochs=600, method="weighted", lbd=0.8),
        dict(patch_size=120, overlap=0, dropout=0.2, k_steps=6, lr_encoder=1e-4, lr_others=5e-5,
             alpha0=8.0, alpha1=2.0, alpha2=0.2, epochs=450, method="mean", lbd=0.35),
        dict(patch_size=32, overlap=25, dropout=0.65, k_steps=2, lr_encoder=5e-5, lr_others=1e-4,
             alpha0=2.0, alpha1=1.0, alpha2=0.05, epochs=300, method="mean", lbd=0.65),
    ]
    for t in tail_configs:
        cfg = ANCHOR.__dict__.copy()
        cfg.update(t)
        add(TrialConfig(**cfg))

    # Preserve the build order, but trim if requested
    if max_trials is not None:
        queue = queue[:max_trials]
    return queue


def run_streaming_command(
    cmd: List[str],
    cwd: Optional[Path],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    retry_label: str = "",
) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as logf:
        log(f"Running command{retry_label}: {' '.join(cmd)}", logf)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip("\n"), logf)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
        log("Command finished successfully.", logf)


def latest_mat_file(folder: Path) -> Optional[Path]:
    mats = sorted(folder.rglob("*.mat"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return mats[-1] if mats else None


def read_auc_from_mat(mat_path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        data = sio.loadmat(str(mat_path))
    except Exception:
        return None, None
    roc = data.get("roc_auc", None)
    pr = data.get("pr_auc", None)

    def scalar(x):
        if x is None:
            return None
        try:
            import numpy as np
            arr = np.asarray(x).squeeze()
            if arr.size == 1:
                return float(arr.item())
            return None
        except Exception:
            return None

    return scalar(roc), scalar(pr)


def safe_relpath(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def trial_dir_for(root: Path, trial_id: str) -> Path:
    return root / "trials" / trial_id


def best_dir_for(root: Path, group: str, train_name: str, test_name: str) -> Path:
    return root / "best" / group / f"{train_name}__to__{test_name}"


def pair_key(group: str, train_name: str, test_name: str) -> str:
    return f"{group}/{train_name}__to__{test_name}"


def update_best_pair(
    state: Dict[str, Any],
    root: Path,
    group: str,
    train_name: str,
    test_name: str,
    auc: float,
    pr_auc: Optional[float],
    trial_id: str,
    params: Dict[str, Any],
    src_mat: Path,
    src_json: Path,
) -> bool:
    key = pair_key(group, train_name, test_name)
    best_pairs = state.setdefault("best_pairs", {})
    prev = best_pairs.get(key, {})
    prev_auc = prev.get("auc", float("-inf"))
    if auc > prev_auc:
        dest = best_dir_for(root, group, train_name, test_name)
        ensure_dir(dest)
        dst_mat = dest / src_mat.name
        dst_json = dest / "meta.json"
        shutil.copy2(src_mat, dst_mat)
        shutil.copy2(src_json, dst_json)
        best_pairs[key] = {
            "auc": auc,
            "pr_auc": pr_auc,
            "trial_id": trial_id,
            "mat_path": str(dst_mat),
            "json_path": str(dst_json),
            "params": params,
            "updated_at": ts(),
        }
        return True
    return False


def compute_trial_mean(pair_results: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [r.get("auc") for r in pair_results if isinstance(r.get("auc"), (int, float))]
    prs = [r.get("pr_auc") for r in pair_results if isinstance(r.get("pr_auc"), (int, float))]
    mean_auc = sum(vals) / len(vals) if vals else None
    mean_pr = sum(prs) / len(prs) if prs else None
    return mean_auc, mean_pr, len(vals)


def main() -> int:
    parser = argparse.ArgumentParser(description="Priority-ordered HADGAN hyperparameter search with resumable state.")
    parser.add_argument("--python-exe", type=str, default=PYTHON_EXE, help="Python executable to run train/infer.")
    parser.add_argument("--train-script", type=str, default=TRAIN_SCRIPT, help="Path to train.py")
    parser.add_argument("--infer-script", type=str, default=INFER_SCRIPT, help="Path to infer.py")
    parser.add_argument("--work-dir", type=str, default=str(WORK_DIR), help="Output root for state and artifacts")
    parser.add_argument("--porcine-dir", type=str, default=PORCINE_DIR, help="Folder holding porcine datasets")
    parser.add_argument("--bovine-dir", type=str, default=BOVINE_DIR, help="Folder holding bovine datasets")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on number of hyperparameter configs")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned queue and exit")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)
    ensure_dir(work_dir / "logs")
    ensure_dir(work_dir / "trials")
    ensure_dir(work_dir / "best")

    state_path = work_dir / "state.json"
    state = load_json(state_path, default={
        "created_at": ts(),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "completed_trial_ids": [],
        "trial_records": {},
        "best_pairs": {},
        "best_overall": {
            "trial_id": None,
            "mean_auc": None,
            "mean_pr_auc": None,
            "updated_at": None,
        },
        "datasets": {},
    })

    porcine = find_mat_datasets(args.porcine_dir)
    bovine = find_mat_datasets(args.bovine_dir)

    groups = {
        "porcine": porcine,
        "bovine": bovine,
    }

    # Store dataset inventory in state
    state["datasets"] = {
        g: [safe_relpath(p) for p in paths] for g, paths in groups.items()
    }
    atomic_write_json(state_path, state)

    all_tasks = []
    for group, paths in groups.items():
        tasks = paired_tasks(group, paths)
        all_tasks.extend(tasks)

    if not all_tasks:
        log("No .mat datasets found in either root. Nothing to do.")
        return 1

    queue = build_priority_queue(args.max_trials)
    if not queue:
        log("No hyperparameter configurations generated.")
        return 1

    if args.dry_run:
        log(f"Dry run: {len(queue)} trial configs, {len(all_tasks)} ordered train/test pairs.")
        for i, cfg in enumerate(queue[:20], 1):
            log(f"  {i:03d}: {cfg.to_dict()}")
        return 0

    # Determine how many trials are already completed
    completed_ids = set(state.get("completed_trial_ids", []))
    total_trials = len(queue)
    completed_trials = sum(1 for cfg in queue if hash_config(cfg.to_dict()) in completed_ids)
    remaining_trials = total_trials - completed_trials

    log(f"Discovered {len(porcine)} porcine and {len(bovine)} bovine datasets.")
    log(f"Planned tasks: {len(all_tasks)} ordered train/test pairs.")
    log(f"Planned hyperparameter configs: {total_trials} (completed so far: {completed_trials}; remaining: {remaining_trials}).")

    # Main trial loop
    for trial_index, cfg in enumerate(queue, 1):
        cfg_dict = cfg.to_dict()
        trial_id = hash_config(cfg_dict)
        trial_path = trial_dir_for(work_dir, trial_id)
        ensure_dir(trial_path)
        ensure_dir(trial_path / "logs")
        ensure_dir(trial_path / "checkpoints")
        ensure_dir(trial_path / "inference")
        trial_log = trial_path / "trial.log"

        if trial_id in completed_ids and (trial_path / "trial_summary.json").exists():
            continue

        remaining_trials = total_trials - sum(1 for c in queue[:trial_index-1] if hash_config(c.to_dict()) in completed_ids)
        remaining_pct = 100.0 * remaining_trials / total_trials if total_trials else 0.0

        with trial_log.open("a", encoding="utf-8") as logf:
            log(f"Starting trial {trial_index}/{total_trials} | remaining configs: {remaining_trials} ({remaining_pct:.2f}%)", logf)
            log(f"Trial ID: {trial_id}", logf)
            log(f"Hyperparameters: {json.dumps(cfg_dict, sort_keys=True)}", logf)

        # Per-trial state record
        trial_state = state.setdefault("trial_records", {}).setdefault(trial_id, {
            "trial_id": trial_id,
            "config": cfg_dict,
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "pair_results": [],
            "mean_auc": None,
            "mean_pr_auc": None,
            "successful_pairs": 0,
            "total_pairs": len(all_tasks),
        })
        if trial_state.get("status") == "completed" and trial_state.get("pair_results"):
            completed_ids.add(trial_id)
            continue
        if trial_state.get("started_at") is None:
            trial_state["started_at"] = ts()
        trial_state["status"] = "running"
        atomic_write_json(state_path, state)

        pair_results: List[Dict[str, Any]] = trial_state.get("pair_results", [])
        pair_results_by_key = {r["pair_key"]: r for r in pair_results if "pair_key" in r}

        try:
            # We train once per train dataset inside each trial, then infer on every other dataset.
            for group, paths in groups.items():
                dataset_names = [normalize_dataset_name(p) for p in paths]
                for train_path in paths:
                    train_name = normalize_dataset_name(train_path)
                    train_out = trial_path / "checkpoints" / group / train_name
                    ensure_dir(train_out)

                    # Train only if not already done for this trial/dataset
                    train_key = f"{group}/{train_name}"
                    if not (train_out.exists() and any(train_out.rglob("ckpt*"))):
                        cmd_train = [
                            args.python_exe,
                            str(Path(args.train_script)),
                            "--hsi_path", str(train_path),
                            "--output_dir", str(train_out),
                            "--patch_size", str(cfg.patch_size),
                            "--epochs", str(cfg.epochs),
                            "--overlap", str(cfg.overlap),
                            "--dropout", str(cfg.dropout),
                            "--k_steps", str(cfg.k_steps),
                            "--seed", str(cfg.seed),
                            "--alpha0", str(cfg.alpha0),
                            "--alpha1", str(cfg.alpha1),
                            "--alpha2", str(cfg.alpha2),
                            "--lr_encoder", str(cfg.lr_encoder),
                            "--lr_others", str(cfg.lr_others),
                        ]
                        train_log = trial_path / "logs" / f"{group}__{train_name}__train.log"
                        success = False
                        last_err = None
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                run_streaming_command(cmd_train, cwd=None, log_path=train_log, retry_label=f" (attempt {attempt}/{MAX_RETRIES})")
                                success = True
                                break
                            except Exception as e:
                                last_err = e
                                with train_log.open("a", encoding="utf-8") as logf:
                                    log(f"Train attempt {attempt} failed for {group}/{train_name}: {e}", logf)
                                time.sleep(2)
                        if not success:
                            raise RuntimeError(f"Training failed for {group}/{train_name}") from last_err

                    # Infer on every other dataset in the same group
                    for test_path in paths:
                        test_name = normalize_dataset_name(test_path)
                        if test_path == train_path:
                            continue

                        pk = pair_key(group, train_name, test_name)
                        if pk in pair_results_by_key and pair_results_by_key[pk].get("status") == "done":
                            continue

                        pair_out = trial_path / "inference" / group / train_name / test_name
                        ensure_dir(pair_out)
                        cmd_infer = [
                            args.python_exe,
                            str(Path(args.infer_script)),
                            "--hsi_path", str(test_path),
                            "--checkpoint_path", str(train_out),
                            "--output_dir", str(pair_out),
                            "--patch_size", str(cfg.patch_size),
                            "--overlap", str(cfg.overlap),
                            "--method", cfg.method,
                            "--lbd", str(cfg.lbd),
                        ]
                        infer_log = pair_out / "infer.log"
                        success = False
                        last_err = None
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                run_streaming_command(cmd_infer, cwd=None, log_path=infer_log, retry_label=f" (attempt {attempt}/{MAX_RETRIES})")
                                success = True
                                break
                            except Exception as e:
                                last_err = e
                                with infer_log.open("a", encoding="utf-8") as logf:
                                    log(f"Inference attempt {attempt} failed for {pk}: {e}", logf)
                                time.sleep(2)
                        if not success:
                            raise RuntimeError(f"Inference failed for {pk}") from last_err

                        result_mat = latest_mat_file(pair_out)
                        if result_mat is None:
                            raise FileNotFoundError(f"No .mat result found in {pair_out}")

                        roc_auc, pr_auc = read_auc_from_mat(result_mat)
                        if roc_auc is None:
                            raise ValueError(f"Could not read {PRIMARY_METRIC} from {result_mat}")

                        pair_meta = {
                            "pair_key": pk,
                            "group": group,
                            "train_dataset": train_name,
                            "test_dataset": test_name,
                            "trial_id": trial_id,
                            "trial_config": cfg_dict,
                            "status": "done",
                            "result_mat": str(result_mat),
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                            "created_at": ts(),
                            "train_output_dir": str(train_out),
                            "infer_output_dir": str(pair_out),
                            "commands": {
                                "train": cmd_train,
                                "infer": cmd_infer,
                            },
                        }
                        pair_json = pair_out / "result_meta.json"
                        atomic_write_json(pair_json, pair_meta)

                        improved = update_best_pair(
                            state=state,
                            root=work_dir,
                            group=group,
                            train_name=train_name,
                            test_name=test_name,
                            auc=roc_auc,
                            pr_auc=pr_auc,
                            trial_id=trial_id,
                            params=cfg_dict,
                            src_mat=result_mat,
                            src_json=pair_json,
                        )

                        pair_results_by_key[pk] = pair_meta
                        pair_results = list(pair_results_by_key.values())
                        trial_state["pair_results"] = pair_results
                        trial_state["successful_pairs"] = len([r for r in pair_results if r.get("status") == "done"])
                        atomic_write_json(state_path, state)

                        with trial_log.open("a", encoding="utf-8") as logf:
                            if improved:
                                log(f"NEW BEST for {pk}: {PRIMARY_METRIC}={roc_auc:.6f} | copied to best store", logf)
                            else:
                                log(f"{pk}: {PRIMARY_METRIC}={roc_auc:.6f} | pr_auc={pr_auc if pr_auc is not None else 'n/a'}", logf)

            mean_auc, mean_pr_auc, n_ok = compute_trial_mean(list(pair_results_by_key.values()))
            trial_state["mean_auc"] = mean_auc
            trial_state["mean_pr_auc"] = mean_pr_auc
            trial_state["successful_pairs"] = n_ok
            trial_state["finished_at"] = ts()
            trial_state["status"] = "completed"
            atomic_write_json(trial_path / "trial_summary.json", trial_state)

            # Update global best-overall trial by mean AUC
            best_overall = state.setdefault("best_overall", {
                "trial_id": None,
                "mean_auc": None,
                "mean_pr_auc": None,
                "updated_at": None,
            })
            prev_mean = best_overall.get("mean_auc")
            if mean_auc is not None and (prev_mean is None or mean_auc > prev_mean):
                best_overall.update({
                    "trial_id": trial_id,
                    "mean_auc": mean_auc,
                    "mean_pr_auc": mean_pr_auc,
                    "updated_at": ts(),
                    "config": cfg_dict,
                    "trial_summary": str(trial_path / "trial_summary.json"),
                })

            completed_ids.add(trial_id)
            state["completed_trial_ids"] = sorted(completed_ids)
            atomic_write_json(state_path, state)

            with trial_log.open("a", encoding="utf-8") as logf:
                log(f"Trial completed: mean_{PRIMARY_METRIC}={mean_auc:.6f} over {n_ok} successful ordered pairs." if mean_auc is not None else
                    "Trial completed, but no successful pairs were scored.", logf)
                if mean_pr_auc is not None:
                    log(f"Trial mean_{SECONDARY_METRIC}={mean_pr_auc:.6f}", logf)
                log(f"Global best mean AUC so far: {state['best_overall'].get('mean_auc')}", logf)

        except Exception as e:
            trial_state["status"] = "partial"
            trial_state["finished_at"] = ts()
            trial_state["error"] = f"{type(e).__name__}: {e}"
            trial_state["traceback"] = traceback.format_exc()
            atomic_write_json(trial_path / "trial_summary.json", trial_state)
            atomic_write_json(state_path, state)
            with trial_log.open("a", encoding="utf-8") as logf:
                log(f"Trial {trial_id} interrupted/failed: {e}", logf)
                log("State has been saved; rerun this script to resume.", logf)
            # keep going to next trial rather than aborting the whole sweep
            continue

    # Final write
    atomic_write_json(state_path, state)
    log("Search run finished.")
    if state.get("best_overall", {}).get("trial_id") is not None:
        log(f"Best overall trial: {state['best_overall']['trial_id']} | mean_{PRIMARY_METRIC}={state['best_overall'].get('mean_auc')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
