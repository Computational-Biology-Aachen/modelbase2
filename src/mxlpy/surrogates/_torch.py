from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.optimizer import ParamsT

from mxlpy.nn._torch import MLP, DefaultDevice, train
from mxlpy.types import AbstractSurrogate

type LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

__all__ = [
    "LossFn",
    "Torch",
    "TorchTrainer",
    "train_torch",
]


def _mean_abs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard loss for surrogates.

    Args:
        x: Predictions of a model.
        y: Targets.

    Returns:
        torch.Tensor: loss.

    """
    return torch.mean(torch.abs(x - y))


@dataclass(kw_only=True)
class Torch(AbstractSurrogate):
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


@dataclass(init=False)
class TorchTrainer:
    features: pd.DataFrame
    targets: pd.DataFrame
    approximator: nn.Module
    optimizer: Adam
    device: torch.device
    losses: list[pd.Series]
    loss_fn: LossFn

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        approximator: nn.Module | None = None,
        optimimzer_cls: Callable[[ParamsT], Adam] = Adam,
        device: torch.device = DefaultDevice,
        loss_fn: LossFn = _mean_abs,
    ) -> None:
        self.features = features
        self.targets = targets

        if approximator is None:
            approximator = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[50, 50, len(targets.columns)],
            )
        self.approximator = approximator.to(device)

        self.optimizer = optimimzer_cls(approximator.parameters())
        self.device = device
        self.loss_fn = loss_fn
        self.losses = []

    def train(
        self,
        epochs: int,
        batch_size: int | None = None,
    ) -> Self:
        losses = train(
            aprox=self.approximator,
            features=self.features.to_numpy(),
            targets=self.targets.to_numpy(),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            device=self.device,
            loss_fn=self.loss_fn,
        )
        if len(self.losses) > 0:
            losses.index += self.losses[-1].index[-1]
        self.losses.append(losses)
        return self

    def get_loss(self) -> pd.Series:
        return pd.concat(self.losses)

    def get_surrogate(
        self,
        surrogate_args: list[str] | None = None,
        surrogate_outputs: list[str] | None = None,
        surrogate_stoichiometries: dict[str, dict[str, float]] | None = None,
    ) -> Torch:
        return Torch(
            model=self.approximator,
            args=surrogate_args if surrogate_args is not None else [],
            outputs=surrogate_outputs if surrogate_outputs is not None else [],
            stoichiometries=surrogate_stoichiometries
            if surrogate_stoichiometries is not None
            else {},
        )


def train_torch(
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
    loss_fn: LossFn = _mean_abs,
) -> tuple[Torch, pd.Series]:
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
        ...)surrogate_stoichiometries

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
        loss_fn: Custom loss function or instance of torch loss object

    Returns:
        tuple[TorchSurrogate, pd.Series]: Trained surrogate model and loss history.

    """
    trainer = TorchTrainer(
        features=features,
        targets=targets,
        approximator=approximator,
        optimimzer_cls=optimimzer_cls,
        device=device,
        loss_fn=loss_fn,
    ).train(
        epochs=epochs,
        batch_size=batch_size,
    )
    return trainer.get_surrogate(
        surrogate_args=surrogate_args,
        surrogate_outputs=surrogate_outputs,
        surrogate_stoichiometries=surrogate_stoichiometries,
    ), trainer.get_loss()
