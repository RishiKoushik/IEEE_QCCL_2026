# src/data/generate_sparse_paulis.py

import numpy as np
import itertools

PAULI_SINGLE = {
    0: np.array([[1, 0], [0, 1]], dtype=np.complex128),   # I
    1: np.array([[0, 1], [1, 0]], dtype=np.complex128),   # X
    2: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),# Y
    3: np.array([[1, 0], [0, -1]], dtype=np.complex128),  # Z
}

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def generate_sparse_paulis(n_qubits, save_path):
    sparse_paulis = []

    for term in itertools.product([0,1,2,3], repeat=n_qubits):
        mats = [PAULI_SINGLE[t] for t in term]
        P = kron_all(mats)

        vec = P.reshape(-1)
        idx = np.nonzero(vec)[0]
        val = vec[idx]

        sparse_paulis.append((term, idx, val))

    np.savez_compressed(
        save_path,
        sparse_paulis=np.array(sparse_paulis, dtype=object),
        n_qubits=n_qubits
    )

if __name__ == "__main__":
    generate_sparse_paulis(
        n_qubits=5,
        save_path="data/paulis/sparse_paulis_5q.npz"
    )
