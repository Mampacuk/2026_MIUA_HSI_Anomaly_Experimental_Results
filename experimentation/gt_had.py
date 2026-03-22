import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------
# Edit these constants.
# ---------------------------------------------------------------------
MAIN_PY = r"C:\Users\alexa\PycharmProjects\GT-HAD\dnnmethods\GT-HAD\main.py"
INPUT_DIR = r"D:\dev\experimental_results\experimentation\input\input130x174"
WORK_DIR = r"D:\dev\experimental_results\experimentation\output\gt_had_output"
CUDA_VISIBLE_DEVICES = "0"

# The script evaluates every candidate on every .mat file in INPUT_DIR.
# It keeps a resumable checkpoint in WORK_DIR/checkpoint.json and
# stores the best heatmaps in WORK_DIR/best/.
# ---------------------------------------------------------------------

CHECKPOINT_JSON = Path(WORK_DIR) / "checkpoint.json"
TRIALS_DIR = Path(WORK_DIR) / "trials"
BEST_DIR = Path(WORK_DIR) / "best"
BEST_REPORT_JSON = Path(WORK_DIR) / "best_report.json"
PYTHON_BIN = r"D:\dev\IEEE_JSTARS_NL2Net\.venv\Scripts\python.exe"
# Paper anchor:
# - 150 epochs
# - batch size 64
# - Adam lr 1e-3
# - patch_size=3, patch_stride=3
# - embed_dim=64
# - mlp_ratio=2.0
# - attn_drop=0.0
# - drop=0.0
# - residual diffusion kernel 3x3x5 in paper notation
#
# For biomedical lesions that are larger than typical remote-sensing anomalies,
# the most useful sweep is cube geometry + width + learning rate.
#
# Geometry candidates are expressed as (patch_size, patch_stride).
DEFAULTS = {
    "patch_size": 3,
    "patch_stride": 3,
    "block_stride": 3,
    "embed_dim": 64,
    "mlp_ratio": 2.0,
    "attn_drop": 0.0,
    "drop": 0.0,
    "lr": 1e-3,
    "batch_size": 64,
    "end_iter": 150,
    "search_iter": 25,
    "rd_bands": 5,
    "rd_rows": 3,
    "rd_cols": 3,
}

GEOMETRY_GRID = [
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (5, 3),
]

EMBED_DIM_GRID = [48, 64, 96, 128]
LR_GRID = [1e-3, 3e-4, 1e-4]

# previously fixed, now varied
BLOCK_STRIDE_GRID = [2, 3]
MLP_RATIO_GRID = [1.5, 2.0, 3.0]
ATTN_DROP_GRID = [0.0, 0.05]
DROP_GRID = [0.0, 0.05]
END_ITER_GRID = [100, 150, 200]
SEARCH_ITER_GRID = [15, 25, 50]
RD_BANDS_GRID = [3, 5, 7]

def normalize_config(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg)
    return out

def load_checkpoint() -> dict:
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    return {
        "trials": {},
        "best": None,
    }


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(CHECKPOINT_JSON)


def canonical_config(cfg: dict) -> str:
    cfg = normalize_config(cfg)
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))

def config_id(cfg: dict) -> str:
    return sha1(canonical_config(cfg).encode("utf-8")).hexdigest()[:12]


def migrate_checkpoint(state: dict) -> dict:
    state.setdefault("trials", {})
    for k, trial in state["trials"].items():
        if "config" in trial:
            trial["config"] = normalize_config(trial["config"])
            trial["key"] = config_id(trial["config"])
    if state.get("best") and "config" in state["best"]:
        state["best"]["config"] = normalize_config(state["best"]["config"])
        state["best"]["key"] = config_id(state["best"]["config"])
    return state

def list_datasets(input_dir: str) -> List[Path]:
    p = Path(input_dir)
    mats = sorted([x for x in p.iterdir() if x.suffix.lower() == ".mat"])
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")
    return mats


def candidate_configs():
    for patch_size, patch_stride in GEOMETRY_GRID:
        for embed_dim in EMBED_DIM_GRID:
            for lr in LR_GRID:
                for block_stride in BLOCK_STRIDE_GRID:
                    for mlp_ratio in MLP_RATIO_GRID:
                        for attn_drop in ATTN_DROP_GRID:
                            for drop in DROP_GRID:
                                for end_iter in END_ITER_GRID:
                                    for search_iter in SEARCH_ITER_GRID:
                                        for rd_bands in RD_BANDS_GRID:
                                            yield normalize_config({
                                                "patch_size": patch_size,
                                                "patch_stride": patch_stride,
                                                "embed_dim": embed_dim,
                                                "lr": lr,
                                                "block_stride": block_stride,
                                                "mlp_ratio": mlp_ratio,
                                                "attn_drop": attn_drop,
                                                "drop": drop,
                                                "end_iter": end_iter,
                                                "search_iter": search_iter,
                                                "rd_bands": rd_bands,
                                            })


def trial_dir_for(cfg: dict) -> Path:
    return TRIALS_DIR / f"trial_{config_id(cfg)}"


def best_trial_dir_for(cfg: dict) -> Path:
    return BEST_DIR / f"best_{config_id(cfg)}"


def run_main_on_file(main_py: str, data_dir: str, save_dir: str, stem: str, cfg: dict, report_json: Path) -> dict:
    cmd = [
        PYTHON_BIN,
        main_py,
        "--file", stem,
        "--data-dir", data_dir,
        "--save-dir", save_dir,
        "--patch-size", str(cfg["patch_size"]),
        "--patch-stride", str(cfg["patch_stride"]),
        "--block-stride", str(cfg["block_stride"]),
        "--embed-dim", str(cfg["embed_dim"]),
        "--mlp-ratio", str(cfg["mlp_ratio"]),
        "--attn-drop", str(cfg["attn_drop"]),
        "--drop", str(cfg["drop"]),
        "--lr", str(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--end-iter", str(cfg["end_iter"]),
        "--search-iter", str(cfg["search_iter"]),
        "--rd-bands", str(cfg["rd_bands"]),
        "--rd-rows", str(cfg["rd_rows"]),
        "--rd-cols", str(cfg["rd_cols"]),
        "--report-json", str(report_json),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"main.py failed for {stem} with config {cfg}\n"
            f"---- stdout/stderr ----\n{proc.stdout}"
        )

    if report_json.exists():
        return json.loads(report_json.read_text(encoding="utf-8"))

    # Fallback parser if JSON writing was disabled for some reason.
    auc = None
    for line in proc.stdout.splitlines():
        if line.strip().startswith("Auc:"):
            auc = float(line.split("Auc:", 1)[1].strip())
    if auc is None:
        raise RuntimeError(f"Could not recover AUC for {stem}. Output:\n{proc.stdout}")
    return {"auc": auc, "stdout": proc.stdout}


def ensure_copied_best_outputs(trial_root: Path, best_root: Path, dataset_files: List[Path]) -> None:
    best_root.mkdir(parents=True, exist_ok=True)
    for mat_path in dataset_files:
        stem = mat_path.stem
        src_dir = trial_root / stem
        dst_dir = best_root / stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ["GT-HAD_map.mat", "GT-HAD_roc.mat"]:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)


def run():
    checkpoint = load_checkpoint()
    checkpoint = migrate_checkpoint(checkpoint)
    trials = checkpoint.setdefault("trials", {})
    best = checkpoint.get("best")

    dataset_files = list_datasets(INPUT_DIR)
    dataset_stems = [p.stem for p in dataset_files]

    print(f"Found {len(dataset_files)} dataset(s): {dataset_stems}")
    print(f"Checkpoint: {CHECKPOINT_JSON}")

    for cfg in candidate_configs():
        key = config_id(cfg)
        if key in trials and trials[key].get("status") == "done":
            print(f"Skipping already-completed config {config_id(cfg)}")
            continue

        trial_root = trial_dir_for(cfg)
        trial_root.mkdir(parents=True, exist_ok=True)

        trial_state = trials.get(key, {
            "config": cfg,
            "status": "running",
            "per_file": {},
        })
        trial_state["config"] = cfg
        trial_state["status"] = "running"
        trial_state.setdefault("per_file", {})

        print("\n=== Trial", config_id(cfg), "===")
        print(cfg)

        # Evaluate only files that are not already present in the checkpoint.
        for mat_path in dataset_files:
            stem = mat_path.stem
            if stem in trial_state["per_file"]:
                print(f"  already done: {stem}")
                continue

            report_json = trial_root / stem / "report.json"
            report_json.parent.mkdir(parents=True, exist_ok=True)
            save_dir = str(trial_root)

            print(f"  running {stem} ...")
            try:
                result = run_main_on_file(
                    MAIN_PY, INPUT_DIR, save_dir, stem, cfg, report_json
                )
            except RuntimeError as e:
                print("Subprocess failed, retrying same config...")
                continue  # reattempt same dataset/config
            trial_state["per_file"][stem] = {
                "auc": float(result["auc"]),
                "report": str(report_json),
            }

            # Update partial checkpoint after each file so the run is resumable at any moment.
            partial_auc_values = [v["auc"] for v in trial_state["per_file"].values()]
            trial_state["mean_auc"] = float(sum(partial_auc_values) / len(partial_auc_values))
            trials[key] = trial_state
            checkpoint["trials"] = trials
            save_checkpoint(checkpoint)

            print(f"  {stem}: AUC={result['auc']:.6f}")

        # Mark completed trial.
        auc_values = [v["auc"] for v in trial_state["per_file"].values()]
        if len(auc_values) != len(dataset_files):
            # This can happen only if the run is interrupted mid-trial.
            continue

        mean_auc = float(sum(auc_values) / len(auc_values))
        trial_state["mean_auc"] = mean_auc
        trial_state["status"] = "done"
        trials[key] = trial_state

        print(f"Mean AUC for trial {config_id(cfg)}: {mean_auc:.6f}")

        # New best?
        if best is None or mean_auc > best["mean_auc"]:
            print("  -> new best config")
            best = {
                "key": key,
                "config": cfg,
                "mean_auc": mean_auc,
                "per_file": trial_state["per_file"],
                "trial_root": str(trial_root),
            }
            checkpoint["best"] = best

            # Copy the candidate's artifacts into the best folder, overwriting the previous best.
            if BEST_DIR.exists():
                shutil.rmtree(BEST_DIR)
            ensure_copied_best_outputs(trial_root, BEST_DIR, dataset_files)
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            BEST_REPORT_JSON.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

        checkpoint["trials"] = trials
        checkpoint["best"] = best
        save_checkpoint(checkpoint)

    print("\nSearch finished.")
    if checkpoint.get("best"):
        print("Best mean AUC:", checkpoint["best"]["mean_auc"])
        print("Best config:", checkpoint["best"]["config"])
        print("Best artifacts:", BEST_DIR)
    else:
        print("No completed trials yet.")


if __name__ == "__main__":
    run()
