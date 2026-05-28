"""
Unified QCCL sweep runner.

Usage:
    python sweep.py --model quantum     --run-ids all --workers 8
    python sweep.py --model classical   --run-ids all --dataset mnist
    python sweep.py --model classical_k --run-ids 1-10 --workers 4
    python sweep.py --model quantum     --run-ids all --fresh   # erase existing results first
"""
import argparse
import csv
import fcntl
import itertools
import subprocess
import sys
import time
import os

import numpy as np
import torch

from src.train import train
from src.models.classical import ClassicalConfig
from src.models.qml import QNNConfig

# ============================================================
# SYSTEM THREAD CONTROL
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ============================================================
# SHARED CONFIG
# ============================================================
PAULI_FILE = "data/paulis/sparse_paulis_5q.npz"
DEVICE     = "cpu"

EPOCHS     = 50
BATCH_SIZE = 64
LR         = 5e-3
OPT_STEPS  = 10
N_QUBITS   = 5

SEEDS = [42,4,14]   # extend to [0, 1, 2, ...] for a multi-seed study

# ============================================================
# QUANTUM SWEEP DIMENSIONS
# ============================================================
REPRESENTATIONS = ["flatten", "pauli"]
ANSATZ_TYPES    = ["be", "se", "random", "random_indep"]
KS              = [2, 4, 8, 16, 32]
INIT_STATES     = ["zero", "random_fixed", "random_indep"]

# Param counts per neuron for n_qubits=5:
#   SE:          depth × 5 × 3 = 15 × depth
#   BE / Random: depth × 5     =  5 × depth
#
# Extra depths for BE/Random so they can be param-count matched to SE at 3, 5, 7:
#   SE depth 3  (45 params) ↔  BE/Random depth  9  (45 params)
#   SE depth 5  (75 params) ↔  BE/Random depth 15  (75 params)
#   SE depth 7 (105 params) ↔  BE/Random depth 21 (105 params)
DEPTHS_SE    = [1, 3, 5, 7]
DEPTHS_OTHER = [1, 3, 5, 7, 9, 15, 21]

# ============================================================
# MODEL METADATA
# ============================================================
MODEL_HEADERS = {
    "quantum": [
        "run_id", "seed", "representation", "ansatz", "depth",
        "K", "init_state", "test_accuracy", "runtime_sec",
    ],
    "classical":   ["run_id", "seed", "representation", "K", "test_accuracy", "runtime_sec"],
    "classical_k": ["run_id", "seed", "representation", "K", "test_accuracy", "runtime_sec"],
}


RESULTS_DIR = "results"

def get_out_file(model, dataset):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if model == "quantum":
        return f"{RESULTS_DIR}/results_{dataset}.csv"
    elif model == "classical":
        return f"{RESULTS_DIR}/results_{dataset}_classical_linear.csv"
    elif model == "classical_k":
        return f"{RESULTS_DIR}/results_{dataset}_classical_linear_k.csv"

# ============================================================
# RUN LIST
# ============================================================
def all_runs(model):
    if model == "quantum":
        runs = []
        for seed, rep, ansatz, K, init_state in itertools.product(
            SEEDS, REPRESENTATIONS, ANSATZ_TYPES, KS, INIT_STATES
        ):
            depths = DEPTHS_SE if ansatz == "se" else DEPTHS_OTHER
            for depth in depths:
                runs.append((seed, rep, ansatz, depth, K, init_state))
        return runs
    else:
        return list(itertools.product(SEEDS, REPRESENTATIONS, KS))


def run_key(model, config):
    if model == "quantum":
        seed, rep, ansatz, depth, K, init_state = config
        return (int(seed), rep, ansatz, int(depth), int(K), init_state)
    else:
        seed, rep, K = config
        return (int(seed), rep, int(K))


def run_key_from_row(model, row):
    if model == "quantum":
        return (
            int(row["seed"]), row["representation"], row["ansatz"],
            int(row["depth"]), int(row["K"]), row["init_state"],
        )
    else:
        return (int(row["seed"]), row["representation"], int(row["K"]))


def make_csv_row(model, run_id, config, test_acc, runtime):
    if model == "quantum":
        seed, rep, ansatz, depth, K, init_state = config
        return [run_id, seed, rep, ansatz, depth, K, init_state, test_acc, runtime]
    else:
        seed, rep, K = config
        return [run_id, seed, rep, K, test_acc, runtime]


# ============================================================
# TRAINING
# ============================================================
def execute_run(model, config, data_dir):
    if model == "quantum":
        seed, representation, ansatz, depth, K, init_state = config

        rng = np.random.default_rng(seed)
        param_seeds   = rng.integers(0, 2**31 - 1, size=K).tolist()
        arch_seeds    = rng.integers(0, 2**31 - 1, size=K).tolist()
        init_st_seeds = rng.integers(0, 2**31 - 1, size=K).tolist()

        qnn_configs = [
            QNNConfig(
                ansatz_type=ansatz,
                depth=depth,
                param_seed=int(param_seeds[i]),
                arch_seed=(
                    None               if ansatz in ("be", "se") else
                    seed               if ansatz == "random"     else
                    int(arch_seeds[i])                               # random_indep
                ),
                init_state_type=init_state,
                init_state_seed=(
                    None               if init_state == "zero"         else
                    seed               if init_state == "random_fixed" else
                    int(init_st_seeds[i])                              # random_indep
                ),
            )
            for i in range(K)
        ]

        return train(
            data_dir=data_dir,
            representation=representation,
            normalize=True,
            model_type="quantum_linear",
            classical_config=ClassicalConfig(model_type="linear", bias=True),
            qnn_configs=qnn_configs,
            n_qubits=N_QUBITS,
            pauli_file=PAULI_FILE,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=EPOCHS,
            optimizer_steps_per_epoch=OPT_STEPS,
            seed=seed,
            device=DEVICE,
        )

    else:
        seed, representation, K = config
        classical_model_type = "linear" if model == "classical" else "linear_k"

        return train(
            data_dir=data_dir,
            representation=representation,
            normalize=True,
            model_type="classical",
            classical_config=ClassicalConfig(
                model_type=classical_model_type,
                hidden_dim=K,
                bias=True,
            ),
            qnn_configs=None,
            n_qubits=N_QUBITS,
            pauli_file=PAULI_FILE,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=EPOCHS,
            optimizer_steps_per_epoch=OPT_STEPS,
            seed=seed,
            device=DEVICE,
        )


# ============================================================
# CSV HELPERS
# ============================================================
def ensure_csv_header(out_file, header):
    if os.path.exists(out_file):
        return
    with open(out_file, "w", newline="") as f:
        csv.writer(f).writerow(header)


def load_completed(out_file, model):
    completed = set()
    if not os.path.exists(out_file):
        return completed
    with open(out_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(run_key_from_row(model, row))
            except (KeyError, ValueError):
                pass  # skip malformed/legacy rows
    return completed


def append_result_row(out_file, row):
    """Append one result row with an exclusive lock (safe across parallel workers)."""
    with open(out_file, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            csv.writer(f).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ============================================================
# WORKER SPLITTING
# ============================================================
def parse_run_ids(run_str, total_runs):
    if run_str.strip().lower() == "all":
        return set(range(1, total_runs + 1))
    selected = set()
    for part in run_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            selected.update(range(a, b + 1))
        else:
            selected.add(int(part))
    return selected


def split_ids(ids, n):
    ids = sorted(ids)
    return [ids[i::n] for i in range(n) if ids[i::n]]


def ids_to_str(ids):
    return ",".join(str(i) for i in ids)


# ============================================================
# ETA FORMATTING
# ============================================================
def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hr"


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="QCCL sweep runner")
    parser.add_argument(
        "--model", choices=["quantum", "classical", "classical_k"], required=True,
        help="Model type to sweep",
    )
    parser.add_argument(
        "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist",
        help="Dataset to use (default: fashion_mnist)",
    )
    parser.add_argument(
        "--run-ids", type=str, required=True,
        help="Run range (e.g. '1-80'), comma list ('1,5,9'), or 'all'",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Erase existing results for this model/dataset before starting",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    dataset  = args.dataset
    data_dir = f"data/{dataset}"
    out_file = get_out_file(args.model, dataset)
    header   = MODEL_HEADERS[args.model]

    runs       = all_runs(args.model)
    total_runs = len(runs)

    selected_ids = parse_run_ids(args.run_ids, total_runs)
    valid_selected   = sorted(rid for rid in selected_ids if 1 <= rid <= total_runs)
    invalid_selected = [rid for rid in selected_ids if rid < 1 or rid > total_runs]

    if args.fresh and os.path.exists(out_file):
        os.remove(out_file)
        print(f"Erased existing results: {out_file}")

    ensure_csv_header(out_file, header)
    completed = load_completed(out_file, args.model)

    completed_in_sel = sum(
        1 for rid in valid_selected
        if run_key(args.model, runs[rid - 1]) in completed
    )
    remaining = len(valid_selected) - completed_in_sel

    print(f"\n{'='*44}")
    print(f"  SWEEP  [{args.model.upper()}]  —  {dataset}")
    print(f"{'='*44}")
    print(f"Total runs    : {total_runs}")
    print(f"Selected      : {len(valid_selected)}")
    print(f"Already done  : {completed_in_sel}")
    print(f"To run        : {remaining}")
    if invalid_selected:
        print(f"⚠ Invalid IDs : {invalid_selected}")

    if remaining == 0:
        print("Nothing to run. Exiting.")
        return

    print(f"{'='*44}\n")

    # --------------------------------------------------
    # MULTI-WORKER SPAWNING
    # --------------------------------------------------
    if args.workers > 1 and not args.worker:
        pending = [
            rid for rid in valid_selected
            if run_key(args.model, runs[rid - 1]) not in completed
        ]
        chunks = split_ids(pending, args.workers)
        print(f"Spawning {len(chunks)} workers across {len(pending)} pending runs...\n")

        procs = [
            subprocess.Popen([
                sys.executable, __file__,
                "--model",    args.model,
                "--dataset",  dataset,
                "--run-ids",  ids_to_str(chunk),
                "--workers",  "1",
                "--worker",
            ])
            for chunk in chunks
        ]
        for p in procs:
            p.wait()

        print("\nAll workers finished.")
        return

    # --------------------------------------------------
    # EXECUTION LOOP
    # --------------------------------------------------
    runtimes = []

    for run_id, config in enumerate(runs, start=1):
        if run_id not in valid_selected:
            continue

        rkey = run_key(args.model, config)
        if rkey in completed:
            print(f"[{run_id}] Skipped (already done)")
            continue

        if args.model == "quantum":
            seed, rep, ansatz, depth, K, init_state = config
            print(f"\n[RUN {run_id}] seed={seed} | {rep} | {ansatz} | "
                  f"depth={depth} | K={K} | init={init_state}")
        else:
            seed, rep, K = config
            print(f"\n[RUN {run_id}] seed={seed} | {rep} | K={K}")

        t0 = time.time()
        try:
            result   = execute_run(args.model, config, data_dir)
            test_acc = result["test_accuracy"]
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            test_acc = None

        runtime = round(time.time() - t0, 2)
        runtimes.append(runtime)

        if test_acc is not None:
            append_result_row(
                out_file,
                make_csv_row(args.model, run_id, config, test_acc, runtime),
            )

        runs_left = remaining - len(runtimes)
        if runs_left > 0:
            avg_t = sum(runtimes) / len(runtimes)
            eta   = fmt_duration(avg_t * runs_left)
            print(f"  → Acc: {test_acc:.4f} | {runtime}s | ETA: {eta} ({runs_left} left)")
        else:
            print(f"  → Acc: {test_acc:.4f} | {runtime}s")

    print(f"\n{'='*44}")
    print("Sweep finished.")
    print(f"{'='*44}")


if __name__ == "__main__":
    main()
