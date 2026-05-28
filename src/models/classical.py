import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassicalConfig:
    model_type: str          # "linear", "linear_k", "mlp"
    input_dim: int = 1024
    num_classes: int = 10
    hidden_dim: Optional[int] = None   # K
    bias: bool = True

class LinearClassifier(nn.Module):
    """
    Linear baseline:
        x -> Linear(input_dim, num_classes)
    """

    def __init__(self, config: ClassicalConfig):
        super().__init__()

        self.linear = nn.Linear(
            config.input_dim,
            config.num_classes,
            bias=config.bias
        )

    def forward(self, x):
        return self.linear(x)

class LinearBottleneckClassifier(nn.Module):
    """
    Linear bottleneck (parameterization control):
        x -> Linear(input_dim, K) -> Linear(K, num_classes)
    No nonlinearity.
    """

    def __init__(self, config: ClassicalConfig):
        super().__init__()

        assert config.hidden_dim is not None, \
            "hidden_dim (K) must be set for linear_k model"

        self.fc1 = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=config.bias
        )
        self.fc2 = nn.Linear(
            config.hidden_dim,
            config.num_classes,
            bias=config.bias
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MLPClassifier(nn.Module):
    """
    Minimal MLP:
        x -> Linear(input_dim, K) -> ReLU -> Linear(K, num_classes)
    """

    def __init__(self, config: ClassicalConfig):
        super().__init__()

        assert config.hidden_dim is not None, \
            "hidden_dim (K) must be set for mlp model"

        self.fc1 = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=config.bias
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            config.hidden_dim,
            config.num_classes,
            bias=config.bias
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def build_classical_model(config: ClassicalConfig) -> nn.Module:
    if config.model_type == "linear":
        return LinearClassifier(config)

    elif config.model_type == "linear_k":
        return LinearBottleneckClassifier(config)

    elif config.model_type == "mlp":
        return MLPClassifier(config)

    else:
        raise ValueError(f"Unknown classical model type: {config.model_type}")
