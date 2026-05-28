import os
import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from src.data.transforms import mnist_like, cifar_to_gray
from src.data.make_splits import make_train_val_split
from src.data.pauli_pipeline import run_pauli_pipeline

pauli_file = "data/paulis/sparse_paulis_5q.npz"
# -------------------------------------------------
# Helper: save image + pauli splits
# -------------------------------------------------
def save_dataset_with_pauli(
    dataset,
    train_idx,
    val_idx,
    test_dataset,
    out_dir,
    pauli_file,
):
    os.makedirs(out_dir, exist_ok=True)

    def extract(ds, indices=None):
        if indices is not None:
            ds = Subset(ds, indices)
        X = torch.stack([ds[i][0] for i in range(len(ds))]).float()
        y = torch.tensor([ds[i][1] for i in range(len(ds))])
        return X, y

    print("  → Extracting train split")
    X_img_train, y_train = extract(dataset, train_idx)

    print("  → Extracting val split")
    X_img_val, y_val = extract(dataset, val_idx)

    print("  → Extracting test split")
    X_img_test, y_test = extract(test_dataset)

    print("  → Computing Pauli representations (no normalization)")
    X_pauli_train = run_pauli_pipeline(X_img_train, pauli_file)
    X_pauli_val   = run_pauli_pipeline(X_img_val,   pauli_file)
    X_pauli_test  = run_pauli_pipeline(X_img_test,  pauli_file)

    print("  → Saving tensors")

    # images
    torch.save(X_img_train, f"{out_dir}/X_img_train.pt")
    torch.save(X_img_val,   f"{out_dir}/X_img_val.pt")
    torch.save(X_img_test,  f"{out_dir}/X_img_test.pt")

    # pauli (unnormalized)
    torch.save(X_pauli_train, f"{out_dir}/X_pauli_train.pt")
    torch.save(X_pauli_val,   f"{out_dir}/X_pauli_val.pt")
    torch.save(X_pauli_test,  f"{out_dir}/X_pauli_test.pt")

    # labels
    torch.save(y_train, f"{out_dir}/y_train.pt")
    torch.save(y_val,   f"{out_dir}/y_val.pt")
    torch.save(y_test,  f"{out_dir}/y_test.pt")

    print(f"  ✔ Saved dataset to {out_dir}")
    print(f"    Train images: {X_img_train.shape}, Pauli: {X_pauli_train.shape}")
    print(f"    Val images:   {X_img_val.shape}, Pauli: {X_pauli_val.shape}")
    print(f"    Test images:  {X_img_test.shape}, Pauli: {X_pauli_test.shape}")


# -------------------------------------------------
# Dataset-specific preparation
# -------------------------------------------------
def prepare_mnist(root="data", pauli_file=pauli_file):
    print("\n=== Preparing MNIST ===")

    train_ds = MNIST(
        root=f"{root}/raw",
        train=True,
        download=True,
        transform=mnist_like(),
    )
    test_ds = MNIST(
        root=f"{root}/raw",
        train=False,
        download=True,
        transform=mnist_like(),
    )

    train_idx, val_idx = make_train_val_split(len(train_ds))

    save_dataset_with_pauli(
        train_ds,
        train_idx,
        val_idx,
        test_ds,
        out_dir=f"{root}/mnist",
        pauli_file=pauli_file,
    )


def prepare_fashion_mnist(root="data", pauli_file=pauli_file):
    print("\n=== Preparing Fashion-MNIST ===")

    train_ds = FashionMNIST(
        root=f"{root}/raw",
        train=True,
        download=True,
        transform=mnist_like(),
    )
    test_ds = FashionMNIST(
        root=f"{root}/raw",
        train=False,
        download=True,
        transform=mnist_like(),
    )

    train_idx, val_idx = make_train_val_split(len(train_ds))

    save_dataset_with_pauli(
        train_ds,
        train_idx,
        val_idx,
        test_ds,
        out_dir=f"{root}/fashion_mnist",
        pauli_file=pauli_file,
    )


def prepare_cifar10(root="data", pauli_file=pauli_file):
    print("\n=== Preparing CIFAR-10 (grayscale) ===")

    train_ds = CIFAR10(
        root=f"{root}/raw",
        train=True,
        download=True,
        transform=cifar_to_gray(),
    )
    test_ds = CIFAR10(
        root=f"{root}/raw",
        train=False,
        download=True,
        transform=cifar_to_gray(),
    )

    train_idx, val_idx = make_train_val_split(len(train_ds))

    save_dataset_with_pauli(
        train_ds,
        train_idx,
        val_idx,
        test_ds,
        out_dir=f"{root}/cifar10",
        pauli_file=pauli_file,
    )


# -------------------------------------------------
# CLI entry point
# -------------------------------------------------
if __name__ == "__main__":
    print("======================================")
    print(" DATASET PREPARATION PIPELINE STARTED")
    print("======================================")

    prepare_mnist()
    prepare_fashion_mnist()
    # prepare_cifar10()  # available but unused in the paper

    print("\n======================================")
    print(" ALL DATASETS PREPARED SUCCESSFULLY ✅")
    print("======================================")

