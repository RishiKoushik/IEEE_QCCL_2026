import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from tqdm import tqdm

from src.data.pipeline import build_representation
from src.models.classical import ClassicalConfig, build_classical_model
from src.models.qml import QNNConfig
from src.models.quantum_model import build_quantum_model


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def load_split(data_dir: str, split: str, representation: str):
    """
    Load split depending on representation.

    - flatten  -> load image tensors
    - pauli    -> load precomputed Pauli tensors
    """
    if representation == "flatten":
        X = torch.load(f"{data_dir}/X_img_{split}.pt")

    elif representation == "pauli":
        X = torch.load(f"{data_dir}/X_pauli_{split}.pt")

    else:
        raise ValueError(f"Unknown representation: {representation}")

    y = torch.load(f"{data_dir}/y_{split}.pt")
    return X, y


# --------------------------------------------------
# Training function
# --------------------------------------------------
def train(
    *,
    data_dir: str,
    representation: str,
    normalize: bool,
    model_type: str,
    classical_config: ClassicalConfig | None,
    qnn_configs: list[QNNConfig] | None,
    n_qubits: int | None,
    pauli_file: str | None,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 20,
    optimizer_steps_per_epoch: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> Dict:

    set_seed(seed)
    device = torch.device(device)

    # -----------------------
    # Load data
    # -----------------------
    X_train, y_train = load_split(data_dir, "train", representation)
    X_val, y_val     = load_split(data_dir, "val", representation)
    X_test, y_test   = load_split(data_dir, "test", representation)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val     = X_val.to(device), y_val.to(device)
    X_test, y_test   = X_test.to(device), y_test.to(device)

    # -----------------------
    # Build model
    # -----------------------
    if model_type.startswith("classical"):
        model = build_classical_model(classical_config)
    elif model_type.startswith("quantum"):
        model = build_quantum_model(
            qnn_configs=qnn_configs,
            n_qubits=n_qubits,
            pauli_file=pauli_file,
            model_type=model_type,
            num_classes=classical_config.num_classes,
            bias=classical_config.bias,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0])

        total_batches = int(np.ceil(X_train.shape[0] / batch_size))

        if optimizer_steps_per_epoch is None:
            step_points = list(range(1, total_batches + 1))
        else:
            step_points = np.linspace(
                0, total_batches,
                optimizer_steps_per_epoch + 1,
                dtype=int
            )[1:].tolist()

        batch_counter = 0
        step_counter = 0
        loss_accum = None
        loss_accum_count = 0
        epoch_loss = 0.0

        batch_iter = tqdm(
            range(0, X_train.shape[0], batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
        )

        for i in batch_iter:
            idx = perm[i:i + batch_size]
            X_img = X_train[idx]
            y = y_train[idx]

            X = build_representation(
                X_img,
                representation=representation,
                normalize=normalize,
                pauli_file=pauli_file,
            ).to(device)

            logits = model(X)
            loss = criterion(logits, y)

            # Accumulate unscaled batch losses; average over the window before stepping.
            # This keeps the effective gradient magnitude independent of optimizer_steps_per_epoch.
            loss_accum = loss if loss_accum is None else loss_accum + loss
            loss_accum_count += 1
            epoch_loss += loss.item() * X.shape[0]

            batch_counter += 1

            if batch_counter in step_points:
                (loss_accum / loss_accum_count).backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_accum = None
                loss_accum_count = 0
                step_counter += 1

                if hasattr(model, "invalidate_cache"):
                    model.invalidate_cache()

            batch_iter.set_postfix(
                opt_steps=step_counter,
            )

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        with torch.no_grad():
            X_val_rep = build_representation(
                X_val,
                representation=representation,
                normalize=normalize,
                pauli_file=pauli_file,
            ).to(device)

            val_logits = model(X_val_rep)
            val_acc = accuracy(val_logits, y_val)

        # print(
        #     f"Epoch [{epoch+1}/{epochs}] "
        #     f"Loss: {epoch_loss / X_train.shape[0]:.4f} "
        #     f"Val Acc: {val_acc:.4f} "
        #     f"(Optimizer steps: {step_counter})"
        # )

    # -----------------------
    # Test evaluation
    # -----------------------
    model.eval()
    with torch.no_grad():
        X_test_rep = build_representation(
            X_test,
            representation=representation,
            normalize=normalize,
            pauli_file=pauli_file,
        ).to(device)

        test_logits = model(X_test_rep)
        test_acc = accuracy(test_logits, y_test)

    return {
        "model": model,
        "test_accuracy": test_acc,
    }
