from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mxlpy.types import AbstractSurrogate, Array

__all__ = ["QSS", "QSSFn"]

type QSSFn = Callable[..., Array]


@dataclass(kw_only=True)
class QSS(AbstractSurrogate):
    model: QSSFn

    def predict_raw(self, y: Array) -> Array:
        return self.model(y)
