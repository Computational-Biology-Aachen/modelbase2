"""Neural Network Parameter Estimation (NPE) Module.

This module provides classes and functions for training neural network models to estimate
parameters in metabolic models. It includes functionality for both steady-state and
time-series data.

Functions:
    train_torch_surrogate: Train a PyTorch surrogate model
    train_torch_time_course_estimator: Train a PyTorch time course estimator
"""

from __future__ import annotations

__all__ = [
    "AbstractEstimator",
    "DefaultCache",
    "TorchSSEstimator",
    "TorchTimeCourseEstimator",
    "train_torch_ss_estimator",
    "train_torch_time_course_estimator",
]

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam

from modelbase2.nnarchitectures import MLP, DefaultDevice, LSTMnn
from modelbase2.parallel import Cache

DefaultCache = Cache(Path(".cache"))


@dataclass(kw_only=True)
class AbstractEstimator:
    """Abstract class for parameter estimation using neural networks."""

    parameter_names: list[str]

    @abstractmethod
    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""


@dataclass(kw_only=True)
class TorchSSEstimator(AbstractEstimator):
    """Estimator for steady state data using PyTorch models."""

    model: torch.nn.Module

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        with torch.no_grad():
            pred = self.model(torch.tensor(features.to_numpy(), dtype=torch.float32))
            return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass(kw_only=True)
class TorchTimeCourseEstimator(AbstractEstimator):
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


def _train_batched(
    approximator: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    optimizer: Adam,
    batch_size: int,
) -> pd.Series:
    losses = {}

    for epoch in tqdm.trange(epochs):
        permutation = torch.randperm(features.size()[0])
        epoch_loss = 0
        for i in range(0, features.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i : i + batch_size]

            loss = torch.mean(
                torch.abs(approximator(features[indices]) - targets[indices])
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()

        losses[epoch] = epoch_loss / (features.size()[0] / batch_size)
    return pd.Series(losses, dtype=float)


def _train_full(
    approximator: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    optimizer: Adam,
) -> pd.Series:
    losses = {}
    for i in tqdm.trange(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.abs(approximator(features) - targets))
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach().numpy()
    return pd.Series(losses, dtype=float)


def train_torch_ss_estimator(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: type[Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchSSEstimator, pd.Series]:
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
    if approximator is None:
        n_hidden = max(2 * len(features.columns) * len(targets.columns), 10)
        n_outputs = len(targets.columns)
        approximator = MLP(
            n_inputs=len(features.columns), layers=[n_hidden, n_hidden, n_outputs]
        ).to(device)

    features_ = torch.Tensor(features.to_numpy(), device=device)
    targets_ = torch.Tensor(targets.to_numpy(), device=device)

    optimizer = optimimzer_cls(approximator.parameters())
    if batch_size is None:
        losses = _train_full(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
        )
    else:
        losses = _train_batched(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=batch_size,
        )

    return TorchSSEstimator(
        model=approximator,
        parameter_names=list(targets.columns),
    ), losses


def train_torch_time_course_estimator(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    approximator: nn.Module | None = None,
    optimimzer_cls: type[Adam] = Adam,
    device: torch.device = DefaultDevice,
) -> tuple[TorchTimeCourseEstimator, pd.Series]:
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
    if approximator is None:
        approximator = LSTMnn(
            n_inputs=len(features.columns),
            n_outputs=len(targets.columns),
            n_hidden=1,
        ).to(device)

    optimizer = optimimzer_cls(approximator.parameters())
    features_ = torch.Tensor(
        np.swapaxes(
            features.to_numpy().reshape((len(targets), -1, len(features.columns))),
            axis1=0,
            axis2=1,
        ),
        device=device,
    )
    targets_ = torch.Tensor(targets.to_numpy(), device=device)
    if batch_size is None:
        losses = _train_full(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
        )
    else:
        losses = _train_batched(
            approximator=approximator,
            features=features_,
            targets=targets_,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=batch_size,
        )
    return TorchTimeCourseEstimator(
        model=approximator,
        parameter_names=list(targets.columns),
    ), losses
