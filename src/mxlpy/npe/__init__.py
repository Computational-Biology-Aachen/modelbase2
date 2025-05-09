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

with contextlib.suppress(ImportError):
    from ._torch import (
        TorchSteadyState,
        TorchSteadyStateTrainer,
        TorchTimeCourse,
        TorchTimeCourseTrainer,
        train_torch_steady_state,
        train_torch_time_course,
    )

__all__ = [
    "TorchSteadyState",
    "TorchSteadyStateTrainer",
    "TorchTimeCourse",
    "TorchTimeCourseTrainer",
    "train_torch_steady_state",
    "train_torch_time_course",
]
