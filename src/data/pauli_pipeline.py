import numpy as np
import torch


def run_pauli_pipeline(X_img, sparse_pauli_file):
    """
    Pauli feature extraction.

    X_img: torch.Tensor [N, 1, 32, 32]
    sparse_pauli_file: path to precomputed sparse Pauli operators

    Returns:
        torch.Tensor [N, D] (real-valued)
    """

    # Convert to numpy
    images = X_img[:, 0].cpu().numpy()  # [N, 32, 32]

    # Hermitian construction (always on)
    sym = 0.5 * (images + images.transpose(0, 2, 1))
    skew = 0.5 * (images - images.transpose(0, 2, 1))
    rho = sym + 1j * skew

    rho_flat = rho.reshape(rho.shape[0], -1).astype(np.complex128)

    # Load sparse Pauli operators
    sparse_paulis = np.load(
        sparse_pauli_file, allow_pickle=True
    )["sparse_paulis"]

    expvals = np.zeros(
        (rho.shape[0], len(sparse_paulis)), dtype=np.float64
    )

    for j, (_, indices, values) in enumerate(sparse_paulis):
        expvals[:, j] = np.dot(rho_flat[:, indices], values).real

    return torch.from_numpy(expvals)
