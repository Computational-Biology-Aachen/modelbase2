from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, fns, mc
from mxlpy.types import (
    McSteadyStates,
    ResponseCoefficientsByPars,
    SteadyStates,
)


@pytest.fixture
def simple_model() -> Model:
    model = Model()
    model.add_parameters({"k1": 1.0, "k2": 2.0})
    model.add_variables({"S": 10.0, "P": 0.0})

    model.add_reaction(
        "v1",
        fn=fns.mass_action_1s,
        args=["S", "k1"],
        stoichiometry={"S": -1.0, "P": 1.0},
    )

    model.add_reaction(
        "v2",
        fn=fns.mass_action_1s,
        args=["P", "k2"],
        stoichiometry={"P": -1.0},
    )

    return model


@pytest.fixture
def mc_to_scan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k1": [1.0, 1.1, 1.2],
            "k2": [2.0, 2.1, 2.2],
        }
    )


@pytest.fixture
def protocol() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": [0.0, 1.0, 2.0],
            "end": [1.0, 2.0, 3.0],
            "k1": [1.0, 1.5, 2.0],
            "k2": [2.0, 2.5, 3.0],
        }
    )


def test_steady_state(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock concentration and flux results
        concs_results = {
            0: pd.DataFrame({"S": [5.0], "P": [5.0]}),
            1: pd.DataFrame({"S": [4.5], "P": [5.5]}),
            2: pd.DataFrame({"S": [4.0], "P": [6.0]}),
        }

        fluxes_results = {
            0: pd.DataFrame({"v1": [5.0], "v2": [10.0]}),
            1: pd.DataFrame({"v1": [4.95], "v2": [11.55]}),
            2: pd.DataFrame({"v1": [4.8], "v2": [13.2]}),
        }

        # Mock the return value of parallelise
        mock_results = {}
        for k in range(3):
            mock_ss = MagicMock()
            mock_ss.variables = concs_results[k]
            mock_ss.fluxes = fluxes_results[k]
            mock_results[k] = mock_ss

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.steady_state(
            simple_model,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert isinstance(result, SteadyStates)
        assert sorted({idx[0] for idx in result.variables.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.variables.index} == {"S", "P"}
        assert sorted({idx[0] for idx in result.fluxes.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.fluxes.index} == {"v1", "v2"}
        assert result.parameters.equals(mc_to_scan)


def test_time_course(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    time_points = np.linspace(0, 10, 5)

    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock time course data
        concs_results = {}
        fluxes_results = {}

        for k in range(3):
            # For each MC parameter set, create a time course for concentrations and fluxes
            concs_results[k] = pd.DataFrame(
                {
                    "S": [10.0 - i * (k + 1) for i in range(5)],
                    "P": [0.0 + i * (k + 1) for i in range(5)],
                },
                index=time_points,
            ).T

            fluxes_results[k] = pd.DataFrame(
                {
                    "v1": [(10.0 - i * (k + 1)) * (k + 1) for i in range(5)],
                    "v2": [(0.0 + i * (k + 1)) * 2 * (k + 1) for i in range(5)],
                },
                index=time_points,
            ).T

        # Mock the return value of parallelise
        mock_results = {}
        for k in range(3):
            mock_tc = MagicMock()
            mock_tc.variables = concs_results[k]
            mock_tc.fluxes = fluxes_results[k]
            mock_results[k] = mock_tc

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.time_course(
            simple_model,
            time_points=time_points,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert sorted({idx[0] for idx in result.variables.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.variables.index} == {"S", "P"}
        assert sorted({idx[0] for idx in result.fluxes.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.fluxes.index} == {"v1", "v2"}
        assert result.parameters.equals(mc_to_scan)


def test_time_course_over_protocol(
    simple_model: Model, mc_to_scan: pd.DataFrame, protocol: pd.DataFrame
) -> None:
    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock time course data
        concs_results = {}
        fluxes_results = {}

        for k in range(3):
            # For each MC parameter set, create a time course for concentrations and fluxes
            time_points = np.linspace(0, 3, 9)  # 3 time steps, 3 points each

            concs_results[k] = pd.DataFrame(
                {
                    "S": [10.0 - i * (k + 1) for i in range(9)],
                    "P": [0.0 + i * (k + 1) for i in range(9)],
                },
                index=time_points,
            ).T

            fluxes_results[k] = pd.DataFrame(
                {
                    "v1": [(10.0 - i * (k + 1)) * (k + 1) for i in range(9)],
                    "v2": [(0.0 + i * (k + 1)) * 2 * (k + 1) for i in range(9)],
                },
                index=time_points,
            ).T

        # Mock the return value of parallelise
        mock_results = {}
        for k in range(3):
            mock_tc = MagicMock()
            mock_tc.variables = concs_results[k]
            mock_tc.fluxes = fluxes_results[k]
            mock_results[k] = mock_tc

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.time_course_over_protocol(
            simple_model,
            protocol=protocol,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert sorted({idx[0] for idx in result.variables.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.variables.index} == {"S", "P"}
        assert sorted({idx[0] for idx in result.fluxes.index}) == [0, 1, 2]
        assert {idx[1] for idx in result.fluxes.index} == {"v1", "v2"}
        assert result.parameters.equals(mc_to_scan)
        assert result.protocol.equals(protocol)


def test_scan_steady_state(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    scan_parameters = pd.DataFrame(
        {
            "k1": [0.5, 1.0, 1.5],
        }
    )

    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock scan results
        concs_results = {}
        fluxes_results = {}

        for k in range(3):  # For each MC parameter set
            # Create a dataframe with index being the scanned parameter values
            concs_results[k] = pd.DataFrame(
                {
                    "S": [8.0 - i * (k + 1) for i in range(3)],
                    "P": [2.0 + i * (k + 1) for i in range(3)],
                },
                index=scan_parameters["k1"],
            ).T

            fluxes_results[k] = pd.DataFrame(
                {
                    "v1": [(8.0 - i * (k + 1)) * (0.5 + i * 0.5) for i in range(3)],
                    "v2": [(2.0 + i * (k + 1)) * 2 for i in range(3)],
                },
                index=scan_parameters["k1"],
            ).T

        # Mock the return value of parallelise
        mock_results = {}
        for k in range(3):
            mock_ss = MagicMock()
            mock_ss.variables = concs_results[k]
            mock_ss.fluxes = fluxes_results[k]
            mock_results[k] = mock_ss

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.scan_steady_state(
            simple_model,
            to_scan=scan_parameters,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert isinstance(result, McSteadyStates)
        assert (
            result.variables.index.nlevels == 2
        )  # MC parameter index and scan parameter index
        assert result.fluxes.index.nlevels == 2
        assert result.parameters.equals(scan_parameters)
        assert result.mc_to_scan.equals(mc_to_scan)


def test_variable_elasticities(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    variables = ["S", "P"]
    concs = {"S": 5.0, "P": 5.0}

    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock elasticity results
        mock_results = {
            0: pd.DataFrame({"S": [1.0, 0.0], "P": [0.0, 1.0]}, index=["v1", "v2"]),
            1: pd.DataFrame({"S": [1.1, 0.0], "P": [0.0, 1.1]}, index=["v1", "v2"]),
            2: pd.DataFrame({"S": [1.2, 0.0], "P": [0.0, 1.2]}, index=["v1", "v2"]),
        }

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.variable_elasticities(
            simple_model,
            to_scan=variables,
            variables=concs,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert isinstance(result, pd.DataFrame)
        assert result.index.nlevels == 2  # MC parameter index and reaction index
        assert list(result.columns) == variables


def test_parameter_elasticities(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    parameters = ["k1", "k2"]
    concs = {"S": 5.0, "P": 5.0}

    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock elasticity results
        mock_results = {
            0: pd.DataFrame({"k1": [1.0, 0.0], "k2": [0.0, 1.0]}, index=["v1", "v2"]),
            1: pd.DataFrame({"k1": [1.1, 0.0], "k2": [0.0, 1.1]}, index=["v1", "v2"]),
            2: pd.DataFrame({"k1": [1.2, 0.0], "k2": [0.0, 1.2]}, index=["v1", "v2"]),
        }

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.parameter_elasticities(
            simple_model,
            to_scan=parameters,
            variables=concs,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert isinstance(result, pd.DataFrame)
        assert result.index.nlevels == 2  # MC parameter index and reaction index
        assert list(result.columns) == parameters


def test_response_coefficients(simple_model: Model, mc_to_scan: pd.DataFrame) -> None:
    parameters = ["k1", "k2"]

    with patch("mxlpy.mc.parallelise") as mock_parallelise:
        # Create mock response coefficient results
        concs_results = {
            0: pd.DataFrame({"S": [-1.0, 0.0], "P": [1.0, -1.0]}, index=parameters),
            1: pd.DataFrame({"S": [-1.1, 0.0], "P": [1.1, -1.1]}, index=parameters),
            2: pd.DataFrame({"S": [-1.2, 0.0], "P": [1.2, -1.2]}, index=parameters),
        }

        fluxes_results = {
            0: pd.DataFrame({"v1": [0.0, 0.0], "v2": [1.0, 0.0]}, index=parameters),
            1: pd.DataFrame({"v1": [0.0, 0.0], "v2": [1.1, 0.0]}, index=parameters),
            2: pd.DataFrame({"v1": [0.0, 0.0], "v2": [1.2, 0.0]}, index=parameters),
        }

        # Mock the return value of parallelise
        mock_results = {}
        for k in range(3):
            mock_rc = MagicMock()
            mock_rc.variables = concs_results[k]
            mock_rc.fluxes = fluxes_results[k]
            mock_results[k] = mock_rc

        mock_parallelise.return_value = mock_results

        # Call the function
        result = mc.response_coefficients(
            simple_model,
            to_scan=parameters,
            mc_to_scan=mc_to_scan,
        )

        # Verify the results
        assert isinstance(result, ResponseCoefficientsByPars)
        assert (
            result.variables.index.nlevels == 2
        )  # MC parameter index and parameter index
        assert list(result.variables.columns) == ["S", "P"]
        assert result.fluxes.index.nlevels == 2
        assert list(result.fluxes.columns) == ["v1", "v2"]
        assert result.parameters.equals(mc_to_scan)
