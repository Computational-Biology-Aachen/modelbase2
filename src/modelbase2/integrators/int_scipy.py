"""Scipy integrator for solving ODEs."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Scipy",
]

import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.integrate as spi

from modelbase2.types import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Scipy:
    """Scipy integrator for solving ODEs.

    Attributes:
        rhs: Right-hand side function of the ODE.
        y0: Initial conditions.
        atol: Absolute tolerance for the solver.
        rtol: Relative tolerance for the solver.
        t0: Initial time point.
        _y0_orig: Original initial conditions.

    Methods:
        __post_init__: Initialize the Scipy integrator.
        reset: Reset the integrator.
        integrate: Integrate the ODE system.
        integrate_to_steady_state: Integrate the ODE system to steady state.

    """

    rhs: Callable
    y0: ArrayLike
    atol: float = 1e-8
    rtol: float = 1e-8
    t0: float = 0.0
    _y0_orig: ArrayLike = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create copy of initial state.

        This method creates a copy of the initial state `y0` and stores it in the `_y0_orig` attribute.
        This is useful for preserving the original initial state for future reference or reset operations.

        """
        self._y0_orig = self.y0.copy()

    def reset(self) -> None:
        """Reset the integrator."""
        self.t0 = 0
        self.y0 = self._y0_orig.copy()

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        """Integrate the ODE system.

        Args:
            t_end: Terminal time point for the integration.
            steps: Number of steps for the integration.
            time_points: Array of time points for the integration.

        Returns:
            tuple[ArrayLike | None, ArrayLike | None]: Tuple containing the time points and the integrated values.

        """
        if time_points is not None:
            if time_points[0] != 0:
                t = [self.t0]
                t.extend(time_points)
            else:
                t = cast(list, time_points)
            t_array = np.array(t)
        elif steps is not None and t_end is not None:
            # Scipy counts the total amount of return points rather than
            # steps as assimulo
            steps += 1
            t_array = np.linspace(self.t0, t_end, steps)
        elif t_end is not None:
            t_array = np.linspace(self.t0, t_end, 100)
        else:
            msg = "You need to supply t_end (+steps) or time_points"
            raise ValueError(msg)

        y = spi.odeint(
            func=self.rhs,
            y0=self.y0,
            t=t_array,
            tfirst=True,
            atol=self.atol,
            rtol=self.rtol,
        )
        self.t0 = t_array[-1]
        self.y0 = y[-1, :]
        return t_array, y

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        step_size: int = 100,
        max_steps: int = 1000,
        integrator: str = "lsoda",
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the ODE system to steady state.

        Args:
            tolerance: Tolerance for determining steady state.
            rel_norm: Whether to use relative normalization.
            step_size: Step size for the integration (default: 100).
            max_steps: Maximum number of steps for the integration (default: 1,000).
            integrator: Name of the integrator to use (default: "lsoda").

        Returns:
            tuple[float | None, ArrayLike | None]: Tuple containing the final time point and the integrated values at steady state.

        """
        self.reset()
        integ = spi.ode(self.rhs)
        integ.set_integrator(
            name=integrator,
            step_size=step_size,
            max_steps=max_steps,
            integrator=integrator,
        )
        integ.set_initial_value(self.y0)
        t = self.t0 + step_size
        y1 = copy.deepcopy(self.y0)
        for _ in range(max_steps):
            y2 = integ.integrate(t)
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1
            if np.linalg.norm(diff, ord=2) < tolerance:
                return t, cast(ArrayLike, y2)
            y1 = y2
            t += step_size
        return None, None
