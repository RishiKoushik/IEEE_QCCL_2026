from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import pennylane as qml


@dataclass
class QNNConfig:
    ansatz_type: str          # "be", "se", "random", "random_indep"
    depth: int
    param_seed: int
    init_state_type: str = "zero"
    init_state_seed: Optional[int] = None
    # Architecture seed for random/random_indep ansatze.
    # be/se:         unused — architecture is deterministic from n_qubits + depth.
    # random:        same value for all K neurons → shared gate layout per global seed.
    # random_indep:  distinct value per neuron → K different gate layouts per global seed.
    arch_seed: Optional[int] = None


def prepare_initial_state(n_qubits: int, init_state_type: str, seed: Optional[int]):
    if init_state_type == "zero":
        return None  # |0>^n is PennyLane default

    if init_state_type in ("random_fixed", "random_indep"):
        # Both types use the same generation logic; the caller controls whether
        # seed is shared across neurons (random_fixed) or unique per neuron (random_indep).
        rng = np.random.default_rng(seed)
        state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        state /= np.linalg.norm(state)
        return state

    raise ValueError(f"Unknown init_state_type: {init_state_type}")


def apply_ansatz(ansatz_type: str, params, n_qubits: int, depth: int,
                 arch_seed: Optional[int] = None):
    if ansatz_type == "be":
        qml.BasicEntanglerLayers(params, wires=range(n_qubits))

    elif ansatz_type == "se":
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

    elif ansatz_type == "random":
        # All neurons share the same gate layout; arch_seed selects which layout.
        seed = arch_seed if arch_seed is not None else 42
        qml.RandomLayers(params, wires=range(n_qubits), seed=seed)

    elif ansatz_type == "random_indep":
        # Each neuron has its own gate layout; caller must supply a distinct arch_seed.
        qml.RandomLayers(params, wires=range(n_qubits), seed=arch_seed)

    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}")


def get_param_shape(ansatz_type: str, depth: int, n_qubits: int):
    if ansatz_type == "be":
        return qml.BasicEntanglerLayers.shape(depth, n_qubits)

    if ansatz_type == "se":
        return qml.StronglyEntanglingLayers.shape(depth, n_qubits)

    if ansatz_type in ("random", "random_indep"):
        return qml.RandomLayers.shape(depth, n_qubits)

    raise ValueError(f"Unknown ansatz_type: {ansatz_type}")


def build_pauli_observables(sparse_paulis) -> List:
    """Build PennyLane Pauli observables from the sparse Pauli representation."""
    single = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    ops = []
    for term, _, _ in sparse_paulis:
        op = None
        for i, t in enumerate(term):
            p = single[t](i)
            op = p if op is None else op @ p
        ops.append(op)
    return ops


class QuantumNeuron(torch.nn.Module):
    """
    One quantum neuron producing a Pauli expectation vector.
    """

    def __init__(
        self,
        config: QNNConfig,
        n_qubits: int,
        pauli_ops: List,
        device_name: str = "lightning.qubit",
    ):
        super().__init__()

        self.config = config
        self.n_qubits = n_qubits
        self.pauli_ops = pauli_ops  # pre-built and shared across neurons

        # PennyLane device
        self.dev = qml.device(device_name, wires=n_qubits)

        # Parameter shape
        self.param_shape = get_param_shape(
            config.ansatz_type, config.depth, n_qubits
        )

        # Deterministic parameter initialization via local generator (no global seed impact)
        g = torch.Generator()
        g.manual_seed(config.param_seed)
        self.params = torch.nn.Parameter(
            torch.randn(self.param_shape, generator=g)
        )

        # Initial state
        self.init_state = prepare_initial_state(
            n_qubits,
            config.init_state_type,
            config.init_state_seed,
        )

        # QNode
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch"
        )

    def _circuit(self):
        if self.init_state is not None:
            qml.StatePrep(self.init_state, wires=range(self.n_qubits))

        apply_ansatz(
            self.config.ansatz_type,
            self.params,
            self.n_qubits,
            self.config.depth,
            arch_seed=self.config.arch_seed,
        )

        return [qml.expval(op) for op in self.pauli_ops]

    def forward(self):
        return torch.stack(self.qnode())


class QuantumLayer(torch.nn.Module):
    """
    Quantum-parameterized linear layer:
        z = W(theta) x + b
    """

    def __init__(
        self,
        qnn_configs: List[QNNConfig],
        n_qubits: int,
        pauli_file: str,
        use_bias: bool = True,
    ):
        super().__init__()

        # Load Pauli file and build observables once; all neurons share the same list.
        sparse_paulis = np.load(pauli_file, allow_pickle=True)["sparse_paulis"]
        pauli_ops = build_pauli_observables(sparse_paulis)

        self.qnns = torch.nn.ModuleList(
            [
                QuantumNeuron(cfg, n_qubits, pauli_ops)
                for cfg in qnn_configs
            ]
        )

        self.use_bias = use_bias
        if use_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(len(qnn_configs))
            )

    def debug_seed_sanity(self):
        print("\n=== SEED SANITY CHECK ===")

        for i, qnn in enumerate(self.qnns):
            params_flat = qnn.params.detach().cpu().flatten()
            print(f"Neuron {i}:")
            print("  Param mean:", params_flat.mean().item())
            print("  Param std :", params_flat.std().item())
            print("  First 5 params:", params_flat[:5].tolist())
            print("  arch_seed:", qnn.config.arch_seed)

            if qnn.init_state is not None:
                print("  Init state first 3 amplitudes:",
                      qnn.init_state[:3])
            else:
                print("  Init state: |0>")

        print("=== END SEED CHECK ===\n")

    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            W : torch.Tensor [K, 4^n]
            b : torch.Tensor [K] or None
        """
        W = torch.stack([qnn() for qnn in self.qnns])
        return W, self.bias if self.use_bias else None
