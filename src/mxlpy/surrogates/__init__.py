"""Surrogate Models Module.

This module provides classes and functions for creating and training surrogate models
for metabolic simulations. It includes functionality for both steady-state and time-series
data using neural networks.

Classes:
    AbstractSurrogate: Abstract base class for surrogate models.
    TorchSurrogate: Surrogate model using PyTorch.
    Approximator: Neural network approximator for surrogate modeling.

Functions:
    train_torch_surrogate: Train a PyTorch surrogate model.
    train_torch_time_course_estimator: Train a PyTorch time course estimator.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from ._torch import TorchSurrogate, TorchSurrogateTrainer, train_torch_surrogate

from ._poly import PolySurrogate, train_polynomial_surrogate

__all__ = [
    "PolySurrogate",
    "TorchSurrogate",
    "TorchSurrogateTrainer",
    "train_polynomial_surrogate",
    "train_torch_surrogate",
]
