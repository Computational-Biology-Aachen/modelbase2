"""Collection of neural network architectures."""

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    if TYPE_CHECKING:
        from . import _keras as keras
        from . import _torch as torch
    else:
        from lazy_import import lazy_module

        keras = lazy_module("mxlpy.nn._keras")
        torch = lazy_module("mxlpy.nn._torch")


__all__ = [
    "keras",
    "torch",
]
