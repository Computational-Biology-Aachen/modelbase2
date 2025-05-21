"""Neural Process Estimation (NPE) module.

This module provides classes and functions for estimating metabolic processes using
neural networks. It includes functionality for both steady-state and time-course data.

Classes:
    TorchSteadyState: Class for steady-state neural network estimation.
    TorchSteadyStateTrainer: Class for training steady-state neural networks.
    TorchTimeCourse: Class for time-course neural network estimation.
    TorchTimeCourseTrainer: Class for training time-course neural networks.

Functions:
    train_torch_steady_state: Train a PyTorch steady-state neural network.
    train_torch_time_course: Train a PyTorch time-course neural network.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    if TYPE_CHECKING:
        from . import _keras as keras
        from . import _torch as torch
    else:
        from lazy_import import lazy_module

        keras = lazy_module("mxlpy.npe._keras")
        torch = lazy_module("mxlpy.npe._torch")


__all__ = [
    "keras",
    "torch",
]
