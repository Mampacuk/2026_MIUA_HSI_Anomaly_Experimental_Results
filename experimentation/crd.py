import os
import glob
import datetime
from concurrent.futures import ProcessPoolExecutor

import matlab.engine

# ------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------


DATASET_DIR = "input130x174"
OUTPUT_DIR  = "crd_output"

MATLAB_SCRIPT_DIR = "./matlab_source/CRD"   # folder containing runCRD.m

INNER_WINDOWS = {
    "porcine1":  (25, [1.0, 10.0, 100.0]),
    "porcine2":  (25, [1.0, 10.0, 100.0]),
    "porcine3":  (33, [1.0, 10.0, 100.0]),
    "porcine4":  (17, [1.0, 10.0, 100.0]),
    "bovine1":   (53,[1.0, 10.0, 100.0]),
    "bovine2":   (47, [1.0, 10.0, 100.0]),
    "bovine3":   (43, [1.0, 10.0, 100.0]),
    "bovine4":   (29, [1.0, 10.0, 100.0]),
    "bovine5.1": (25, [1.0, 10.0, 100.0]),
    "bovine5.2": (21, [1.0, 10.0, 100.0]),
    "bovine5.3": (27, [1.0, 10.0, 100.0]),
}

OUTER_MULTIPLIERS = [1.125, 1.25, 1.5, 1.75]


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

def log(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} | {msg}", flush=True)


# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------

def nearest_odd(x):
    n = int(round(x))
    if n % 2 == 0:
        n += 1
    return max(3, n)


def find_params(dataset_name):
    for prefix, (inner, lambdas) in INNER_WINDOWS.items():
        if dataset_name.startswith(prefix):
            return inner, lambdas
    raise ValueError(f"No parameters defined for dataset {dataset_name}")


def compute_outer_windows(inner):
    outs = []

    for m in OUTER_MULTIPLIERS:
        val = nearest_odd(inner * m)

        if val <= inner:
            val = inner + 2

        outs.append(val)

    return sorted(set(outs))


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def process_dataset(mat_path):

    name = os.path.splitext(os.path.basename(mat_path))[0]

    log(f"[START] {name}")

    # start MATLAB engine in this process
    eng = matlab.engine.start_matlab()

    eng.addpath(eng.genpath(MATLAB_SCRIPT_DIR), nargout=0)

    inner, lambdas = find_params(name)
    outers = compute_outer_windows(inner)

    for outer in outers:

        for lam in lambdas:

            out_name = f"CRD_{name}_in{inner}_out{outer}_lam{lam:g}.mat"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            if os.path.exists(out_path):
                log(f"[SKIP] {out_name}")
                continue

            log(f"[RUN ] {name} in={inner} out={outer} λ={lam}")

            eng.runCRD(
                mat_path,
                out_path,
                float(inner),
                float(outer),
                float(lam),
                nargout=0
            )

            log(f"[SAVE] {out_name}")

    eng.quit()

    log(f"[DONE] {name}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mat")))

    if not mat_files:
        raise RuntimeError("No .mat files found")

    log(f"Found {len(mat_files)} datasets")

    with ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(process_dataset, mat_files)


if __name__ == "__main__":
    main()