"""Simulation Module.

This module provides classes and functions for simulating metabolic models.
It includes functionality for running simulations, normalizing results, and
retrieving simulation data.

Classes:
    Simulator: Class for running simulations on a metabolic model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import numpy as np
import pandas as pd

from modelbase2.integrators import DefaultIntegrator

__all__ = ["Simulator"]

if TYPE_CHECKING:
    from modelbase2.model import Model
    from modelbase2.types import ArrayLike, IntegratorProtocol


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    """Normalize split results by a given factor or array.

    Args:
        results: List of DataFrames containing the results to normalize.
        normalise: Normalization factor or array.

    Returns:
        list[pd.DataFrame]: List of normalized DataFrames.

    """
    if isinstance(normalise, int | float):
        return [i / normalise for i in results]
    if len(normalise) == len(results):
        return [(i.T / j).T for i, j in zip(results, normalise, strict=True)]

    results = []
    start = 0
    end = 0
    for i in results:
        end += len(i)
        results.append(i / np.reshape(normalise[start:end], (len(i), 1)))  # type: ignore
        start += end
    return results


@dataclass(
    init=False,
    slots=True,
    eq=False,
)
class Simulator:
    """Simulator class for running simulations on a metabolic model.

    Attributes:
        model: Model instance to simulate.
        y0: Initial conditions for the simulation.
        integrator: Integrator protocol to use for the simulation.
        concs: List of DataFrames containing concentration results.
        args: List of DataFrames containing argument values.
        simulation_parameters: List of dictionaries containing simulation parameters.

    """

    model: Model
    y0: ArrayLike
    integrator: IntegratorProtocol
    concs: list[pd.DataFrame] | None
    args: list[pd.DataFrame] | None
    simulation_parameters: list[dict[str, float]] | None

    def __init__(
        self,
        model: Model,
        y0: dict[str, float] | None = None,
        integrator: type[IntegratorProtocol] = DefaultIntegrator,
        *,
        test_run: bool = True,
    ) -> None:
        """Initialize the Simulator.

        Args:
            model (Model): The model to be simulated.
            y0 (dict[str, float] | None, optional): Initial conditions for the model variables.
                If None, the initial conditions are obtained from the model. Defaults to None.
            integrator (type[IntegratorProtocol], optional): The integrator to use for the simulation.
                Defaults to DefaultIntegrator.
            test_run (bool, optional): If True, performs a test run to ensure the model's methods
                (get_full_concs, get_fluxes, get_right_hand_side) work correctly with the initial conditions.
                Defaults to True.

        """
        self.model = model
        y0 = model.get_initial_conditions() if y0 is None else y0
        self.y0 = [y0[k] for k in model.get_variable_names()]

        self.integrator = integrator(self.model, y0=self.y0)
        self.concs = None
        self.args = None
        self.simulation_parameters = None

        if test_run:
            y0 = dict(zip(model.get_variable_names(), self.y0, strict=True))
            self.model.get_full_concs(y0, 0)
            self.model.get_fluxes(y0, 0)
            self.model.get_right_hand_side(y0, 0)

    def _save_simulation_results(
        self,
        *,
        results: pd.DataFrame,
        skipfirst: bool,
    ) -> None:
        """Save simulation results.

        Args:
            results: DataFrame containing the simulation results.
            skipfirst: Whether to skip the first row of results.

        """
        if self.concs is None:
            self.concs = [results]
        elif skipfirst:
            self.concs.append(results.iloc[1:, :])
        else:
            self.concs.append(results)

        if self.simulation_parameters is None:
            self.simulation_parameters = []
        self.simulation_parameters.append(self.model.parameters)

    def clear_results(self) -> None:
        """Clear simulation results."""
        self.concs = None
        self.args = None
        self.simulation_parameters = None
        if self.integrator is not None:
            self.integrator.reset()

    def simulate(
        self,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> Self:
        """Simulate the model.

        Examples:
            >>> Simulator(model).simulate(t_end=100)
            >>> Simulator(model).simulate(t_end=100, steps=100)
            >>> Simulator(model).simulate(time_points=[0, 10, 20])
            >>> Simulator(model, y0).simulate(t_end=100)

        You can either supply only a terminal time point, or additionally also the
        number of steps or exact time points for which values should be returned.

        Args:
            t_end: Terminal time point for the simulation.
            steps: Number of steps for the simulation.
            time_points: Exact time points for which values should be returned.

        Returns:
            Self: The Simulator instance with updated results.

        """
        if steps is not None and time_points is not None:
            warnings.warn(
                """
            You can either specify the steps or the time return points.
            I will use the time return points""",
                stacklevel=1,
            )
            if t_end is None:
                t_end = time_points[-1]
            time, results = self.integrator.integrate(
                t_end=t_end,
                time_points=time_points,
            )
        elif time_points is not None:
            time, results = self.integrator.integrate(
                t_end=time_points[-1],
                time_points=time_points,
            )
        elif steps is not None:
            if t_end is None:
                msg = "t_end must no be None"
                raise ValueError(msg)
            time, results = self.integrator.integrate(
                t_end=t_end,
                steps=steps,
            )
        else:
            time, results = self.integrator.integrate(
                t_end=t_end,
            )

        if time is None or results is None:
            return self

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds()
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            results,
            index=time,
            columns=self.model.get_variable_names(),
        )
        self._save_simulation_results(results=results_df, skipfirst=True)
        return self

    def simulate_to_steady_state(
        self,
        tolerance: float = 1e-6,
        *,
        rel_norm: bool = False,
    ) -> Self:
        """Simulate the model to steady state.

        Examples:
            >>> Simulator(model).simulate_to_steady_state()
            >>> Simulator(model).simulate_to_steady_state(tolerance=1e-8)
            >>> Simulator(model).simulate_to_steady_state(rel_norm=True)

        You can either supply only a terminal time point, or additionally also the
        number of steps or exact time points for which values should be returned.

        Args:
            tolerance: Tolerance for the steady-state calculation.
            rel_norm: Whether to use relative norm for the steady-state calculation.

        Returns:
            Self: The Simulator instance with updated results.

        """
        time, results = self.integrator.integrate_to_steady_state(
            tolerance=tolerance,
            rel_norm=rel_norm,
        )
        if time is None or results is None:
            return self

        # NOTE: IMPORTANT!
        # model._get_rhs sorts the return array by model.get_compounds
        # Do NOT change this ordering
        results_df = pd.DataFrame(
            data=[results],
            index=[time],
            columns=self.model.get_variable_names(),
        )
        self._save_simulation_results(results=results_df, skipfirst=False)
        return self

    def simulate_over_protocol(
        self,
        protocol: pd.DataFrame,
        time_points_per_step: int = 10,
    ) -> Self:
        """Simulate the model over a given protocol.

        Examples:
            >>> Simulator(model).simulate_over_protocol(
            ...     protocol,
            ...     time_points_per_step=10
            ... )

        Args:
            protocol: DataFrame containing the protocol.
            time_points_per_step: Number of time points per step.

        Returns:
            The Simulator instance with updated results.

        """
        for t_end, pars in protocol.iterrows():
            t_end = cast(pd.Timedelta, t_end)
            self.model.update_parameters(pars.to_dict())
            self.simulate(t_end.total_seconds(), steps=time_points_per_step)
        return self

    def _get_args_vectorised(
        self,
        concs: list[pd.DataFrame],
        params: list[dict[str, float]],
        *,
        include_readouts: bool = True,
    ) -> list[pd.DataFrame]:
        args: list[pd.DataFrame] = []

        for res, p in zip(concs, params, strict=True):
            self.model.update_parameters(p)
            args.append(
                self.model.get_args_time_course(
                    concs=res,
                    include_readouts=include_readouts,
                )
            )
        return args

    @overload
    def get_concs(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> None | list[pd.DataFrame]: ...

    @overload
    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> None | pd.DataFrame: ...

    @overload
    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True] = True,
    ) -> None | pd.DataFrame: ...

    def get_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> None | pd.DataFrame | list[pd.DataFrame]:
        """Get the concentration results.

        Examples:
            >>> Simulator(model).get_concs()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        Returns:
            pd.DataFrame: DataFrame of concentrations.

        """
        if self.concs is None:
            return None

        results = self.concs.copy()
        if normalise is not None:
            results = _normalise_split_results(results=results, normalise=normalise)
        if concatenated:
            return pd.concat(results, axis=0)

        return results

    @overload
    def get_full_concs(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
        include_readouts: bool = True,
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    @overload
    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | None: ...

    def get_full_concs(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
        include_readouts: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get the full concentration results, including derived quantities.

        Examples:
            >>> Simulator(model).get_full_concs()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        Returns: DataFrame of full concentrations.

        """
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if (args := self.args) is None:
            args = self._get_args_vectorised(concs, params)

        names = (
            self.model.get_variable_names() + self.model.get_derived_variable_names()
        )
        if include_readouts:
            names.extend(self.model.get_readout_names())
        full_concs = [i.loc[:, names] for i in args]
        if normalise is not None:
            full_concs = _normalise_split_results(
                results=full_concs,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(full_concs, axis=0)
        return full_concs

    @overload
    def get_fluxes(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame] | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame | None: ...

    @overload
    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | None: ...

    def get_fluxes(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame] | None:
        """Get the flux results.

        Examples:
            >>> Simulator(model).get_fluxes()
            Time             v1         v2
            0.000000   1.000000   10.00000
            0.000100   0.999900   9.999000
            0.000200   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

        """
        if (concs := self.concs) is None:
            return None
        if (params := self.simulation_parameters) is None:
            return None
        if (args := self.args) is None:
            args = self._get_args_vectorised(concs, params)

        fluxes: list[pd.DataFrame] = []
        for y, p in zip(args, params, strict=True):
            self.model.update_parameters(p)
            fluxes.append(self.model.get_fluxes_time_course(args=y))

        if normalise is not None:
            fluxes = _normalise_split_results(
                results=fluxes,
                normalise=normalise,
            )
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    def get_concs_and_fluxes(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Get the concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_concs_and_fluxes()
            (concs, fluxes)


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple of concentrations and fluxes.

        """
        return self.get_concs(), self.get_fluxes()

    def get_full_concs_and_fluxes(
        self,
        *,
        include_readouts: bool = True,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Get the full concentrations and fluxes.

        >>> Simulator(model).get_full_concs_and_fluxes()
        (full_concs, full_fluxes)

        Args:
            include_readouts: Whether to include readouts in the results.

        Returns:
            Full concentrations and fluxes

        """
        return (
            self.get_full_concs(include_readouts=include_readouts),
            self.get_fluxes(),
        )

    def get_results(self) -> pd.DataFrame | None:
        """Get the combined results of concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_results()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   1.000000
            0.000100   0.999900   0.999900   0.999900   0.999900
            0.000200   0.999800   0.999800   0.999800   0.999800

        Returns:
            pd.DataFrame: Combined DataFrame of concentrations and fluxes.

        """
        c, v = self.get_concs_and_fluxes()
        if c is None or v is None:
            return None
        return pd.concat((c, v), axis=1)

    def get_full_results(self) -> pd.DataFrame | None:
        """Get the combined full results of concentrations and fluxes.

        Examples:
            >>> Simulator(model).get_full_results()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   1.000000
            0.000100   0.999900   0.999900   0.999900   0.999900
            0.000200   0.999800   0.999800   0.999800   0.999800

        """
        c, v = self.get_full_concs_and_fluxes()
        if c is None or v is None:
            return None
        return pd.concat((c, v), axis=1)

    def get_new_y0(self) -> dict[str, float] | None:
        """Get the new initial conditions after the simulation.

        Examples:
            >>> Simulator(model).get_new_y0()
            {"ATP": 1.0, "NADPH": 1.0}

        """
        if (res := self.get_concs()) is None:
            return None
        return dict(res.iloc[-1])

    def update_parameter(self, parameter: str, value: float) -> Self:
        """Updates the value of a specified parameter in the model.

        Examples:
            >>> Simulator(model).update_parameter("k1", 0.1)

        Args:
            parameter: The name of the parameter to update.
            value: The new value to set for the parameter.

        """
        self.model.update_parameter(parameter, value)
        return self

    def update_parameters(self, parameters: dict[str, float]) -> Self:
        """Updates the model parameters with the provided dictionary of parameters.

        Examples:
            >>> Simulator(model).update_parameters({"k1": 0.1, "k2": 0.2})

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the new parameter values.

        """
        self.model.update_parameters(parameters)
        return self

    def scale_parameter(self, parameter: str, factor: float) -> Self:
        """Scales the value of a specified parameter in the model.

        Examples:
            >>> Simulator(model).scale_parameter("k1", 0.1)

        Args:
            parameter: The name of the parameter to scale.
            factor: The factor by which to scale the parameter.

        """
        self.model.scale_parameter(parameter, factor)
        return self

    def scale_parameters(self, parameters: dict[str, float]) -> Self:
        """Scales the values of specified parameters in the model.

        Examples:
            >>> Simulator(model).scale_parameters({"k1": 0.1, "k2": 0.2})

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the scaling factors.

        """
        self.model.scale_parameters(parameters)
        return self
