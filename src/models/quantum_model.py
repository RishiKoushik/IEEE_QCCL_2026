from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from src.models.qml import QuantumLayer, QNNConfig


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class QuantumModelConfig:
    head_type: str          # "linear" | "mlp"
    num_classes: int = 10
    bias: bool = True


# -------------------------------------------------
# Quantum classifier with caching
# -------------------------------------------------
class QuantumClassifier(nn.Module):
    """
    Quantum-parameterized classifier:

        θ → QNN → W(θ)
        x → x @ W(θ)^T → [ReLU] → Linear → logits

    IMPORTANT:
    - W is cached and recomputed only when θ changes
    """

    def __init__(
        self,
        qnn_configs: List[QNNConfig],
        n_qubits: int,
        pauli_file: str,
        config: QuantumModelConfig,
    ):
        super().__init__()

        self.K = len(qnn_configs)

        # Quantum weight generator
        self.quantum_layer = QuantumLayer(
            qnn_configs=qnn_configs,
            n_qubits=n_qubits,
            pauli_file=pauli_file,
            use_bias=config.bias,
        )

        # print("\n[DEBUG] Running seed sanity check...")
        # self.quantum_layer.debug_seed_sanity() # debug_seed_check.py


        # Cache for quantum weights
        self._cached_W: Optional[torch.Tensor] = None
        self._cached_b: Optional[torch.Tensor] = None

        # Optional nonlinearity
        self.use_relu = config.head_type == "mlp"
        if self.use_relu:
            self.relu = nn.ReLU()

        # Classical head
        self.head = nn.Linear(
            self.K,
            config.num_classes,
            bias=config.bias,
        )

    def invalidate_cache(self):
        """Call this AFTER optimizer.step()."""
        self._cached_W = None
        self._cached_b = None

    def forward(self, x):
        """
        x: torch.Tensor [batch, 1024]
        """

        # -------------------------
        # Quantum weight caching
        # -------------------------
        if self._cached_W is None:
            W, bq = self.quantum_layer()
            self._cached_W = W
            self._cached_b = bq
        else:
            W, bq = self._cached_W, self._cached_b

        # dtype alignment
        W = W.to(x.dtype)
        if bq is not None:
            bq = bq.to(x.dtype)

        # -------------------------
        # Classical computation
        # -------------------------
        z = x @ W.T

        if bq is not None:
            z = z + bq

        if self.use_relu:
            z = self.relu(z)

        logits = self.head(z)
        return logits


# -------------------------------------------------
# Factory
# -------------------------------------------------
def build_quantum_model(
    qnn_configs: List[QNNConfig],
    n_qubits: int,
    pauli_file: str,
    model_type: str,      # "quantum_linear" | "quantum_mlp"
    num_classes: int = 10,
    bias: bool = True,
):
    assert model_type in {"quantum_linear", "quantum_mlp"}

    head_type = "linear" if model_type == "quantum_linear" else "mlp"

    config = QuantumModelConfig(
        head_type=head_type,
        num_classes=num_classes,
        bias=bias,
    )

    return QuantumClassifier(
        qnn_configs=qnn_configs,
        n_qubits=n_qubits,
        pauli_file=pauli_file,
        config=config,
    )
