import os
import glob
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor
from spectral.algorithms.detectors import rx

# ------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------

DATASET_DIR = "lrx_input"   # <-- change
OUTPUT_DIR  = "lrx_output"     # <-- change

# dataset prefix -> inner window size
INNER_WINDOWS = {
    "porcine1": 97,
    "porcine2": 97,
    "porcine3": 135,
    "porcine4": 71,
    "bovine1": 217,
    "bovine2": 185,
    "bovine3": 169,
    "bovine4": 117,
    "bovine5.1": 97,
    "bovine5.2": 87,
    "bovine5.3": 111,

}

OUTER_MULTIPLIERS = [1.125, 1.25, 1.5, 1.75, 2.0]

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------

def nearest_odd(x):
    """Return nearest odd integer >= 3."""
    n = int(round(x))
    if n % 2 == 0:
        n += 1
    return max(3, n)


def find_inner_window(dataset_name):
    """Match dataset prefix to inner window size."""
    for prefix, win in INNER_WINDOWS.items():
        if dataset_name.startswith(prefix):
            return win
    raise ValueError(f"No inner window defined for dataset {dataset_name}")


def compute_outer_windows(inner):
    """Compute odd outer window sizes from multipliers."""
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

    print(f"[START] {name}")

    mat = sio.loadmat(mat_path)
    cube = mat["data"].astype(float)      # HSI cube

    inner = find_inner_window(name)
    outers = compute_outer_windows(inner)

    for outer in outers:

        out_name = f"LRX_{name}_in{inner}_out{outer}.mat"
        out_path = os.path.join(OUTPUT_DIR, out_name)  

        if os.path.exists(out_path):
            print(f"[SKIP] {out_name}")
            continue

        print(f"[RUN ] {name}  in={inner} out={outer}")

        # LRX computation using spectral
        show = rx(cube, window=(inner, outer))

        sio.savemat(out_path, {"show": show})

        print(f"[SAVE] {out_name}")

    print(f"[DONE] {name}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mat")))

    if not mat_files:
        raise RuntimeError("No .mat files found")

    print(f"Found {len(mat_files)} datasets")

    with ProcessPoolExecutor() as executor:
        executor.map(process_dataset, mat_files)


if __name__ == "__main__":
    main()