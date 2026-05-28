import torch
from typing import Literal

from src.data.vector_ops import l2_normalize


Representation = Literal["flatten", "pauli"]


def build_representation(
    X_img: torch.Tensor,
    representation: Representation,
    normalize: bool,
    pauli_file: str | None = None,  # kept for compatibility, unused
) -> torch.Tensor:
    """
    Build input representation.

    Args:
        X_img:
            - if representation == "flatten": [N, 1, 32, 32]
            - if representation == "pauli":   [N, 1024] (precomputed)
        representation: "flatten" | "pauli"
        normalize: whether to apply ℓ₂ normalization
        pauli_file: UNUSED (kept to avoid changing callers)

    Returns:
        torch.Tensor [N, 1024]
    """

    # ----------- Representation -----------
    if representation == "flatten":
        # [N, 1, 32, 32] -> [N, 1024]
        X = X_img.view(X_img.shape[0], -1)

    elif representation == "pauli":
        # Pauli is already precomputed and passed in
        X = X_img

    else:
        raise ValueError(f"Unknown representation: {representation}")

    # ----------- Sanity check -----------
    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D tensor, got {X.shape}")
    if X.shape[1] != 1024:
        raise RuntimeError(f"Expected dim 1024, got {X.shape[1]}")

    # ----------- Optional normalization -----------
    if normalize:
        X = l2_normalize(X)

    return X.float()
