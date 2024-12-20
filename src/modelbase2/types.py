"""Types Module.

This module provides type definitions and utility types for use throughout the project.
It includes type aliases for arrays, numbers, and callable functions, as well as re-exports
of common types from standard libraries.

Classes:
    DerivedFn: Callable type for derived functions.
    Array: Type alias for numpy arrays of float64.
    Number: Type alias for float, list of floats, or numpy arrays.
    Param: Type alias for parameter specifications.
    RetType: Type alias for return types.
    Axes: Type alias for numpy arrays of matplotlib axes.
    ArrayLike: Type alias for numpy arrays or lists of floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Array",
    "ArrayLike",
    "Axes",
    "Derived",
    "IntegratorProtocol",
    "McSteadyStates",
    "Number",
    "Param",
    "ProtocolByPars",
    "RateFn",
    "Reaction",
    "Readout",
    "ResponseCoefficients",
    "ResponseCoefficientsByPars",
    "RetType",
    "SteadyStates",
    "TimeCourseByPars",
    "unwrap",
    "unwrap2",
]

# Re-exporting some types here, because their imports have
# changed between Python versions and I have no interest in
# fixing it in every file
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes as Axis
from numpy.typing import NDArray

type RateFn = Callable[..., float]
type Array = NDArray[np.float64]
type Number = float | list[float] | Array

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

Axes = NDArray[Axis]  # type: ignore
ArrayLike = NDArray[np.float64] | list[float]


if TYPE_CHECKING:
    from modelbase2.model import Model


def unwrap[T](el: T | None) -> T:
    """Unwraps an optional value, raising an error if the value is None.

    Args:
        el: The value to unwrap. It can be of type T or None.

    Returns:
        The unwrapped value if it is not None.

    Raises:
        ValueError: If the provided value is None.

    """
    if el is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return el


def unwrap2[T1, T2](tpl: tuple[T1 | None, T2 | None]) -> tuple[T1, T2]:
    """Unwraps a tuple of optional values, raising an error if either of them is None.

    Args:
        tpl: The value to unwrap.

    Returns:
        The unwrapped values if it is not None.

    Raises:
        ValueError: If the provided value is None.

    """
    a, b = tpl
    if a is None or b is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return a, b


class IntegratorProtocol(Protocol):
    """Protocol for numerical integrators."""

    def __init__(
        self,
        rhs: Callable,
        y0: ArrayLike,
    ) -> None:
        """Initialise the integrator."""
        ...

    def reset(self) -> None:
        """Reset the integrator."""
        ...

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        """Integrate the system."""
        ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the system to steady state."""
        ...


@dataclass(slots=True)
class Derived:
    """Container for a derived value."""

    fn: RateFn
    args: list[str]
    math: str | None = None


@dataclass(slots=True)
class Reaction:
    """Container for a reaction."""

    fn: RateFn
    stoichiometry: Mapping[str, float | Derived]
    args: list[str]
    math: str | None = None

    def get_modifiers(self, model: Model) -> list[str]:
        """Get the modifiers of the reaction."""
        include = set(model.variables)
        exclude = set(self.stoichiometry)

        return [k for k in self.args if k in include and k not in exclude]


@dataclass(slots=True)
class Readout:
    """Container for a readout."""

    fn: RateFn
    args: list[str]


@dataclass(slots=True)
class ResponseCoefficients:
    """Container for response coefficients."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class ResponseCoefficientsByPars:
    """Container for response coefficients by parameter."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class SteadyStates:
    """Container for steady states."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the steady states as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class McSteadyStates:
    """Container for Monte Carlo steady states."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    mc_parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the steady states as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)


@dataclass(slots=True)
class TimeCourseByPars:
    """Container for time courses by parameter."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux time courses."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get time courses by name."""
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)


@dataclass(slots=True)
class ProtocolByPars:
    """Container for protocols by parameter."""

    concs: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    protocol: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux protocols."""
        return iter((self.concs, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the protocols as a DataFrame."""
        return pd.concat((self.concs, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get concentration or flux by name."""
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)
