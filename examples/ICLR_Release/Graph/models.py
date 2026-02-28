import torch
import torch.nn as nn
import spikeDE.neuron as snn
from typing import Optional


class SpikingGCN(nn.Module):
    """
    Spiking Graph Convolutional Network implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = False,
        tau: Optional[float] = 0.5,
        tau_learnable: Optional[bool] = False,
        threshold: Optional[float] = 1.0,
    ) -> None:
        super(SpikingGCN, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias)
        self.lif = snn.LIFNeuron(tau, threshold, tau_learnable=tau_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))


class DynamicReactiveSpikingGNN(nn.Module):
    """
    Dynamic Reactive Spiking Graph Neural Network implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = False,
        tau: Optional[float] = 0.5,
        tau_learnable: Optional[bool] = False,
        threshold: Optional[float] = 1.0,
    ):
        super(DynamicReactiveSpikingGNN, self).__init__()

        learnable_threshold = nn.Parameter(
            torch.tensor(threshold, dtype=torch.float32), requires_grad=True
        )
        self.fc = nn.Linear(in_features, out_features, bias)
        self.lif = snn.LIFNeuron(tau, learnable_threshold, tau_learnable=tau_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))
