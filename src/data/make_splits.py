import numpy as np
from pathlib import Path


def make_train_val_split(n_samples, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_val = int(val_frac * n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return train_idx, val_idx


if __name__ == "__main__":
    Path("data/splits").mkdir(parents=True, exist_ok=True)

    datasets = {
        "mnist": 60000,
        "fashion_mnist": 60000,
        "cifar10": 50000,
    }

    for name, n in datasets.items():
        train_idx, val_idx = make_train_val_split(n)

        out = Path(f"data/splits/{name}_train_val.npz")
        np.savez(out, train_idx=train_idx, val_idx=val_idx)

        print(f"{name}: train={len(train_idx)}, val={len(val_idx)}")
