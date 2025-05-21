"""Surrogate Models Module.

This module provides classes and functions for creating and training surrogate models
for metabolic simulations. It includes functionality for both steady-state and time-series
data using neural networks.

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

        keras = lazy_module("mxlpy.surrogates._keras")
        torch = lazy_module("mxlpy.surrogates._torch")


from . import _poly as poly
from . import _qss as qss

__all__ = [
    "keras",
    "poly",
    "qss",
    "torch",
]
