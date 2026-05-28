# QCCL — Quantum-Classical Comparative Learning

Code and results accompanying the IEEE QCCL 2026 paper *"Capacity and Architecture in Quantum-Generated Linear Classifiers"*.

The experiments train ensembles of small quantum circuits (5 qubits) as feature generators followed by a linear classifier, compare them against classical linear baselines on MNIST and Fashion-MNIST, and run a factorial study over five architectural factors (ansatz family, depth, capacity K, initialisation, input representation).

---

## Repository layout

```
.
├── sweep.py                       # Main runner: launches the full factorial sweep
├── src/
│   ├── train.py                   # Unified training loop (quantum + classical)
│   ├── data/
│   │   ├── prepare_dataset.py     # Downloads MNIST / Fashion-MNIST, makes splits, computes Pauli features
│   │   ├── generate_sparse_paulis.py
│   │   ├── pauli_pipeline.py      # Image → density matrix → Tr(ρ·Pⱼ) feature map
│   │   ├── pipeline.py            # Representation builder (flatten / pauli)
│   │   ├── preprocess_images.py
│   │   ├── make_splits.py         # Deterministic 80/20 train/val split
│   │   ├── transforms.py          # Resize to 32×32 + ToTensor
│   │   └── vector_ops.py
│   └── models/
│       ├── qml.py                 # Quantum neuron, QNN configs, PennyLane QNode
│       ├── quantum_model.py       # Quantum ensemble + cached forward
│       └── classical.py           # Linear and bottleneck classifiers
├── data/
│   └── paulis/
│       └── sparse_paulis_5q.npz   # Pre-generated 1024 Pauli operators (5 qubits)
└── results/                       # Six CSVs with all paper results (see below)
```

---

## Results files

All CSVs are seed-replicated (`seed ∈ {42, 4, 14}`), 3 seeds per configuration.

| File | Rows | Description |
|---|---:|---|
| `results_mnist.csv` | 2250 | Quantum sweep on MNIST (750 configs × 3 seeds) |
| `results_fashion_mnist.csv` | 2250 | Quantum sweep on Fashion-MNIST (750 configs × 3 seeds) |
| `results_mnist_classical_linear.csv` | 30 | Classical 1-layer linear baseline (MNIST) |
| `results_mnist_classical_linear_k.csv` | 30 | Classical 2-layer bottleneck baseline (MNIST) |
| `results_fashion_mnist_classical_linear.csv` | 30 | Classical 1-layer linear baseline (Fashion-MNIST) |
| `results_fashion_mnist_classical_linear_k.csv` | 30 | Classical 2-layer bottleneck baseline (Fashion-MNIST) |

**Columns (quantum):** `run_id, seed, representation, ansatz, depth, K, init_state, test_accuracy, runtime_sec`
**Columns (classical):** `run_id, seed, representation, K, test_accuracy, runtime_sec`

---

## Experimental design

| Factor | Levels |
|---|---|
| Dataset | MNIST, Fashion-MNIST |
| Representation | `flatten` (1024-d, image resized to 32×32 and flattened), `pauli` (1024-d Pauli expectation values) |
| Ansatz family | `be` (BasicEntangler), `se` (StronglyEntangling), `random`, `random_indep` |
| Depth | SE: {1, 3, 5, 7} · BE/Random: {1, 3, 5, 7, 9, 15, 21} |
| Capacity K (circuits in ensemble) | {2, 4, 8, 16, 32} |
| Initialisation | `zero`, `random_fixed`, `random_indep` |
| Seeds | {42, 4, 14} (3 seeds per config) |

**Fixed hyperparameters (defined in `sweep.py`):** epochs = 50, batch size = 64, learning rate = 5e-3 (Adam), 10 optimiser steps per epoch (gradient accumulation), 5 qubits, CPU only.

**Data preprocessing:** images are resized from 28×28 to 32×32, scaled to [0, 1] via `ToTensor()`, and either flattened (1024-d) or fed through the Pauli feature map (1024 Pauli expectation values; one per operator).

**Train/val split:** deterministic 80/20 split of the torchvision training set (`val_frac=0.2`, `seed=42` in `make_splits.py`), giving 48 000 train / 12 000 val / 10 000 test for both MNIST and Fashion-MNIST.

---

## Reproducing the results from scratch

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the Pauli operator file

The repo ships `data/paulis/sparse_paulis_5q.npz` (1024 Pauli operators for 5 qubits). To regenerate:

```bash
python -m src.data.generate_sparse_paulis
```

### 3. Download datasets and compute features

Downloads MNIST and Fashion-MNIST via torchvision, applies the deterministic 80/20 train/val split, and precomputes Pauli expectation values for the `pauli` representation.

```bash
python -m src.data.prepare_dataset
```

This populates `data/mnist/` and `data/fashion_mnist/` with `X_img_*.pt`, `X_pauli_*.pt`, and `y_*.pt`.

### 4. Run the full sweep

The factorial sweep is 750 configurations × 3 seeds = 2250 runs per dataset.
The runner supports multi-process parallelism via `--workers`, fault-tolerant CSV append-with-lock, and resumption (already-completed configs are skipped on restart).

```bash
# Quantum sweep (the big one)
python sweep.py --model quantum --dataset mnist          --run-ids all --workers 8
python sweep.py --model quantum --dataset fashion_mnist  --run-ids all --workers 8

# Classical baselines (fast — minutes)
python sweep.py --model classical    --dataset mnist          --run-ids all --workers 4
python sweep.py --model classical_k  --dataset mnist          --run-ids all --workers 4
python sweep.py --model classical    --dataset fashion_mnist  --run-ids all --workers 4
python sweep.py --model classical_k  --dataset fashion_mnist  --run-ids all --workers 4
```

Useful flags:

| Flag | Description |
|---|---|
| `--run-ids 1-100` or `1,5,9` or `all` | Select specific run indices |
| `--workers N` | Spawn N parallel subprocesses (interleaved split of pending IDs) |
| `--fresh` | Erase the existing CSV before starting |

Results are appended to `results/results_<dataset>{,_classical_linear,_classical_linear_k}.csv`.

---

## Compute

The full multi-seed sweep (2250 quantum runs per dataset) was executed on **Google Cloud Compute Engine** using a single **`c2-standard-60`** instance:

- 60 vCPUs (Intel Cascade Lake)
- 240 GB RAM
- 50 GB boot disk, Ubuntu 22.04 LTS
- Region: `asia-south1-a`

Wall time per dataset: ≈ 28–30 hours running 55–58 worker subprocesses in parallel via `sweep.py --workers`.

---

## Citation

If you use this code or the released results, please cite the IEEE QCCL 2026 paper.
The official citation and proceedings link will be added here once available.

---

## Acknowledgements

- The codebase, sweep orchestration, and this README were polished with the help of **[Claude Code](https://claude.com/claude-code)** (Anthropic) — including ANOVA experiment design, multi-seed cloud orchestration, and repository cleanup.
- Quantum circuits implemented with [PennyLane](https://pennylane.ai/).
- Datasets via [torchvision](https://pytorch.org/vision/stable/datasets.html).
- Compute provided by Google Cloud free credits.

---

## License

MIT — see [LICENSE](LICENSE).
