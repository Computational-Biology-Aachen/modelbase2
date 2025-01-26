"""Neural network architectures.

This module provides implementations of neural network architectures used for mechanistic learning.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DefaultDevice", "LSTMnn", "MLP"]

DefaultDevice = torch.device("cpu")


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) for surrogate modeling and neural posterior estimation.

    Attributes:
        net: Sequential neural network model.

    Methods:
        forward: Forward pass through the neural network.

    """

    def __init__(
        self,
        n_inputs: int,
        layers: list[int],
        activation: Callable | None = nn.ReLU(),
        output_activation: Callable | None = None,
    ) -> None:
        """Initializes the MLP with the given number of inputs and list of (hidden) layers.

        Args:
            n_inputs (int): The number of input features.
            n_outputs list(int): A list containing the number of neurons in hidden and output layer.
            activation Callable | None (default nn.ReLU()): The activation function to be applied after each hidden layer
            activation Callable | None (default None): The activation function to be applied after the final (output) layer

        For instance, MLP(10, layers = [50, 50, 10]) initializes a neural network with the following architecture:
        - Linear layer with `n_inputs` inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 10 outputs

        The weights of the linear layers are initialized with a normal distribution
        (mean=0, std=0.1) and the biases are initialized to 0.

        """
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation

        levels = []
        previous_neurons = n_inputs

        for idx, neurons in enumerate(self.layers):
            if idx == (len(self.layers) - 1):
                levels.append(nn.Linear(previous_neurons, neurons))

                if self.output_activation:
                    levels.append(self.output_activation)

            else:
                levels.append(nn.Linear(previous_neurons, neurons))

                if self.activation:
                    levels.append(self.activation)

            previous_neurons = neurons

        self.net = nn.Sequential(*levels)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.net(x)


class LSTMnn(nn.Module):
    """Default LSTM neural network model for time-series approximation."""

    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int) -> None:
        """Initializes the neural network model.

        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output features.
            n_hidden (int): Number of hidden units in the LSTM layer.

        """
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(n_inputs, n_hidden)
        self.to_out = nn.Linear(n_hidden, n_outputs)

        nn.init.normal_(self.to_out.weight, mean=0, std=0.1)
        nn.init.constant_(self.to_out.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        # lstm_out, (hidden_state, cell_state)
        _, (hn, _) = self.lstm(x)
        return cast(torch.Tensor, self.to_out(hn[-1]))  # Use last hidden state
