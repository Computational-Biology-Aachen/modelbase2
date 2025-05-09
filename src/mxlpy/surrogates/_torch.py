from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam
from torch.optim.optimizer import ParamsT

from mxlpy.nn._torch import MLP, DefaultDevice
from mxlpy.types import AbstractSurrogate

__all__ = [
    "TorchSurrogate",
    "train_torch_surrogate",
]


@dataclass(kw_only=True)
class TorchSurrogate(AbstractSurrogate):
    """Surrogate model using PyTorch.

    Attributes:
        model: PyTorch neural network model.

    Methods:
        predict: Predict outputs based on input data using the PyTorch model.

    """

    model: torch.nn.Module

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data using the PyTorch model.

        Args:
            y: Input data as a numpy array.

        Returns:
            dict[str, float]: Dictionary mapping output variable names to predicted values.

        """
        with torch.no_grad():
            return self.model(
                torch.tensor(y, dtype=torch.float32),
            ).numpy()


def _train_batched(
    aprox: nn.Module,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
    batch_size: int,
    loss_fn: Callable,
) -> pd.Series:
    """Train the neural network using mini-batch gradient descent.

    Args:
        aprox: Neural network model to train.
        features: Input features as a tensor.
        targets: Target values as a tensor.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        device: torch device
        batch_size: Size of mini-batches for training.
        loss_fn: Loss function

    Returns:
        pd.Series: Series containing the training loss history.

    """
    rng = np.random.default_rng()
    losses = {}
    for i in tqdm.trange(epochs):
        idxs = rng.choice(features.index, size=batch_size)
        X = torch.Tensor(features.iloc[idxs].to_numpy(), device=device)
        Y = torch.Tensor(targets.iloc[idxs].to_numpy(), device=device)
        optimizer.zero_grad()
        pred = aprox(X)
        loss = loss_fn(pred, Y)  # torch.mean(torch.abs(pred - Y))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def _train_full(
    aprox: nn.Module,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
    loss_fn: Callable,
) -> pd.Series:
    """Train the neural network using full-batch gradient descent.

    Args:
        aprox: Neural network model to train.
        features: Input features as a tensor.
        targets: Target values as a tensor.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        device: Torch device
        loss_fn: Loss function

    Returns:
        pd.Series: Series containing the training loss history.

    """
    X = torch.Tensor(features.to_numpy(), device=device)
    Y = torch.Tensor(targets.to_numpy(), device=device)

    losses = {}
    for i in tqdm.trange(epochs):
        optimizer.zero_grad()
        pred = aprox(X)
        loss = loss_fn(pred, Y)  # torch.mean(torch.abs(pred - Y))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def standard_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard loss for surrogates.

    Args:
        x: Predictions of a model.
        y: Targets.

    Returns:
        torch.Tensor: loss.

    """
    return torch.mean(torch.abs(x - y))


def train_torch_surrogate(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    surrogate_args: list[str] | None = None,
    surrogate_outputs: list[str] | None = None,
    surrogate_stoichiometries: dict[str, dict[str, float]] | None = None,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: Callable[[ParamsT], Adam] = Adam,
    device: torch.device = DefaultDevice,
    loss_func: Callable | None = None,
) -> tuple[TorchSurrogate, pd.Series]:
    """Train a PyTorch surrogate model.

    Examples:
        >>> train_torch_surrogate(
        ...     features,
        ...     targets,
        ...     epochs=100,
        ...     surrogate_inputs=["x1", "x2"],
        ...     surrogate_stoichiometries={
        ...         "v1": {"x1": -1, "x2": 1, "ATP": -1},
        ...     },
        ...)

    Args:
        features: DataFrame containing the input features for training.
        targets: DataFrame containing the target values for training.
        epochs: Number of training epochs.
        surrogate_args: Names of inputs arguments for the surrogate model.
        surrogate_outputs: Names of output arguments from the surrogate.
        surrogate_stoichiometries: Mapping of variables to their stoichiometries
        batch_size: Size of mini-batches for training (None for full-batch).
        approximator: Predefined neural network model (None to use default MLP features-50-50-output).
        optimimzer_cls: Optimizer class to use for training (default: Adam).
        device: Device to run the training on (default: DefaultDevice).
        loss_func: Custom loss function or instance of torch loss object

    Returns:
        tuple[TorchSurrogate, pd.Series]: Trained surrogate model and loss history.

    """
    if approximator is None:
        approximator = MLP(
            n_inputs=len(features.columns),
            neurons_per_layer=[50, 50, len(targets.columns)],
        ).to(device)

    loss_fn = standard_loss if loss_func is None else loss_func

    optimizer = optimimzer_cls(approximator.parameters())
    if batch_size is None:
        losses = _train_full(
            aprox=approximator,
            features=features,
            targets=targets,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
    else:
        losses = _train_batched(
            aprox=approximator,
            features=features,
            targets=targets,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            loss_fn=loss_fn,
        )
    surrogate = TorchSurrogate(
        model=approximator,
        args=surrogate_args if surrogate_args is not None else [],
        outputs=surrogate_outputs if surrogate_outputs is not None else [],
        stoichiometries=surrogate_stoichiometries
        if surrogate_stoichiometries is not None
        else {},
    )
    return surrogate, losses
