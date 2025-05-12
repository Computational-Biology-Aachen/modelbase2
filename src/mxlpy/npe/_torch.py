"""Neural Network Parameter Estimation (NPE) Module.

This module provides classes and functions for training neural network models to estimate
parameters in metabolic models. It includes functionality for both steady-state and
time-series data.

Functions:
    train_torch_surrogate: Train a PyTorch surrogate model
    train_torch_time_course_estimator: Train a PyTorch time course estimator
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.adam import Adam

from mxlpy.nn._torch import LSTM, MLP, DefaultDevice, train
from mxlpy.parallel import Cache
from mxlpy.types import AbstractEstimator

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

DefaultCache = Cache(Path(".cache"))

type LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

__all__ = [
    "DefaultCache",
    "LossFn",
    "TorchSteadyState",
    "TorchSteadyStateTrainer",
    "TorchTimeCourse",
    "TorchTimeCourseTrainer",
    "train_torch_steady_state",
    "train_torch_time_course",
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
class TorchSteadyState(AbstractEstimator):
    """Estimator for steady state data using PyTorch models."""

    model: torch.nn.Module

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        with torch.no_grad():
            pred = self.model(torch.tensor(features.to_numpy(), dtype=torch.float32))
            return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass(kw_only=True)
class TorchTimeCourse(AbstractEstimator):
    """Estimator for time course data using PyTorch models."""

    model: torch.nn.Module

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        idx = cast(pd.MultiIndex, features.index)
        features_ = torch.Tensor(
            np.swapaxes(
                features.to_numpy().reshape(
                    (
                        len(idx.levels[0]),
                        len(idx.levels[1]),
                        len(features.columns),
                    )
                ),
                axis1=0,
                axis2=1,
            ),
        )
        with torch.no_grad():
            pred = self.model(features_)
            return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass
class TorchSteadyStateTrainer:
    """Trainer for steady state data using PyTorch models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    approximator: nn.Module
    optimimzer: Adam
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
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            approximator: Predefined neural network model (None to use default MLP)
            optimimzer_cls: Optimizer class to use for training (default: Adam)
            device: Device to run the training on (default: DefaultDevice)
            loss_fn: Loss function

        """
        self.features = features
        self.targets = targets

        if approximator is None:
            n_hidden = max(2 * len(features.columns) * len(targets.columns), 10)
            n_outputs = len(targets.columns)
            approximator = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[n_hidden, n_hidden, n_outputs],
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
        """Train the model using the provided features and targets.

        Args:
            epochs: Number of training epochs
            batch_size: Size of mini-batches for training (None for full-batch)

        """
        losses = train(
            aprox=self.approximator,
            features=self.features.to_numpy(),
            targets=self.targets.to_numpy(),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        if len(self.losses) > 0:
            losses.index += self.losses[-1].index[-1]
        self.losses.append(losses)
        return self

    def get_loss(self) -> pd.Series:
        """Get the loss history of the training process."""
        return pd.concat(self.losses)

    def get_estimator(self) -> TorchSteadyState:
        """Get the trained estimator."""
        return TorchSteadyState(
            model=self.approximator,
            parameter_names=list(self.targets.columns),
        )


@dataclass
class TorchTimeCourseTrainer:
    """Trainer for time course data using PyTorch models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    approximator: nn.Module
    optimimzer: Adam
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
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            approximator: Predefined neural network model (None to use default LSTM)
            optimimzer_cls: Optimizer class to use for training (default: Adam)
            device: Device to run the training on (default: DefaultDevice)
            loss_fn: Loss function

        """
        self.features = features
        self.targets = targets

        if approximator is None:
            approximator = LSTM(
                n_inputs=len(features.columns),
                n_outputs=len(targets.columns),
                n_hidden=1,
            ).to(device)
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
        """Train the model using the provided features and targets.

        Args:
            epochs: Number of training epochs
            batch_size: Size of mini-batches for training (None for full-batch)

        """
        losses = train(
            aprox=self.approximator,
            features=np.swapaxes(
                self.features.to_numpy().reshape(
                    (len(self.targets), -1, len(self.features.columns))
                ),
                axis1=0,
                axis2=1,
            ),
            targets=self.targets.to_numpy(),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        if len(self.losses) > 0:
            losses.index += self.losses[-1].index[-1]
        self.losses.append(losses)
        return self

    def get_loss(self) -> pd.Series:
        """Get the loss history of the training process."""
        return pd.concat(self.losses)

    def get_estimator(self) -> TorchTimeCourse:
        """Get the trained estimator."""
        return TorchTimeCourse(
            model=self.approximator,
            parameter_names=list(self.targets.columns),
        )


def train_torch_steady_state(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: Callable[[ParamsT], Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchSteadyState, pd.Series]:
    """Train a PyTorch steady state estimator.

    This function trains a neural network model to estimate steady state data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_torch_ss_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        approximator: Predefined neural network model (None to use default MLP)
        optimimzer_cls: Optimizer class to use for training (default: Adam)
        device: Device to run the training on (default: DefaultDevice)

    Returns:
        tuple[TorchTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = TorchSteadyStateTrainer(
        features=features,
        targets=targets,
        approximator=approximator,
        optimimzer_cls=optimimzer_cls,
        device=device,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()


def train_torch_time_course(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: Callable[[ParamsT], Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchTimeCourse, pd.Series]:
    """Train a PyTorch time course estimator.

    This function trains a neural network model to estimate time course data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_torch_time_course_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        approximator: Predefined neural network model (None to use default LSTM)
        optimimzer_cls: Optimizer class to use for training (default: Adam)
        device: Device to run the training on (default: DefaultDevice)

    Returns:
        tuple[TorchTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = TorchTimeCourseTrainer(
        features=features,
        targets=targets,
        approximator=approximator,
        optimimzer_cls=optimimzer_cls,
        device=device,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()
