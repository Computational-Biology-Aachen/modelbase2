from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mxlpy.model import Model
from mxlpy.scan import (
    TimeCourse,
    TimePoint,
    steady_state,
)

if TYPE_CHECKING:
    from mxlpy.types import Array, IntegratorType


def mock_ss_worker(
    model: Model,  # noqa: ARG001
    *,
    rel_norm: bool,  # noqa: ARG001
    integrator: IntegratorType,  # noqa: ARG001
) -> TimePoint:
    return TimePoint(
        variables=pd.Series({"x1": 1.0, "x2": 2.0}),
        fluxes=pd.Series({"v1": 0.1, "v2": 0.2}),
    )


def mock_tc_worker(
    model: Model,  # noqa: ARG001
    time_points: Array,
    integrator: IntegratorType,  # noqa: ARG001
) -> TimeCourse:
    return TimeCourse(
        variables=pd.DataFrame({i: {"x1": 1.0, "x2": 2.0} for i in time_points}).T,
        fluxes=pd.DataFrame({i: {"v1": 0.1, "v2": 0.2} for i in time_points}).T,
    )


def mock_protocol_worker(
    model: Model,  # noqa: ARG001
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,  # noqa: ARG001
) -> TimeCourse:
    return TimeCourse(
        variables=pd.DataFrame({i: {"x1": 1.0, "x2": 2.0} for i in protocol}).T,
        fluxes=pd.DataFrame({i: {"v1": 0.1, "v2": 0.2} for i in protocol}).T,
    )


def test_steady_state_1p() -> None:
    parameters = pd.DataFrame({"param1": [0.1]})
    result = steady_state(
        model=Model().add_parameters({"param1": 0.1}),
        to_scan=parameters,
        worker=mock_ss_worker,
        parallel=False,
    )
    pd.testing.assert_frame_equal(
        result.variables,
        pd.DataFrame(
            {"x1": [1.0], "x2": [2.0]},
            index=pd.Index(parameters["param1"]),
        ),
    )
    pd.testing.assert_frame_equal(
        result.fluxes,
        pd.DataFrame(
            {"v1": [0.1], "v2": [0.2]},
            index=pd.Index(parameters["param1"]),
        ),
    )


def test_steady_state_2p() -> None:
    parameters = pd.DataFrame({"param1": [0.1, 0.2], "param2": [0.3, 0.4]})
    result = steady_state(
        model=Model().add_parameters({"param1": 0.1, "param2": 0.2}),
        to_scan=parameters,
        worker=mock_ss_worker,
        parallel=False,
    )
    pd.testing.assert_frame_equal(
        result.variables,
        pd.DataFrame(
            {"x1": [1.0, 1.0], "x2": [2.0, 2.0]},
            index=pd.MultiIndex.from_frame(parameters),
        ),
    )
    pd.testing.assert_frame_equal(
        result.fluxes,
        pd.DataFrame(
            {"v1": [0.1, 0.1], "v2": [0.2, 0.2]},
            index=pd.MultiIndex.from_frame(parameters),
        ),
    )


# FIXME
# def test_time_course_1p() -> None:
#     parameters = pd.DataFrame({"param1": [0.1]})
#     time_points = np.array([0.0, 1.0])

#     result = time_course(
#         model=Model().add_parameters({"param1": 0.1}),
#         parameters=parameters,
#         time_points=time_points,
#         worker=mock_tc_worker,
#         parallel=False,
#     )
#     pd.testing.assert_frame_equal(
#         result.variables,
#         pd.DataFrame(
#             {
#                 "x1": {(0, 0.0): 1.0, (0, 1.0): 1.0},
#                 "x2": {(0, 0.0): 2.0, (0, 1.0): 2.0},
#             },
#         ),
#     )
#     pd.testing.assert_frame_equal(
#         result.fluxes,
#         pd.DataFrame(
#             {
#                 "v1": {(0, 0.0): 0.1, (0, 1.0): 0.1},
#                 "v2": {(0, 0.0): 0.2, (0, 1.0): 0.2},
#             },
#         ),
#     )


# FIXME
# def test_time_course_over_protocol_1p() -> None:
#     parameters = pd.DataFrame({"param1": [0.1]})
#     protocol = make_protocol([])

#     result = time_course_over_protocol(
#         model=Model().add_parameters({"param1": 0.1}),
#         parameters=parameters,
#         protocol=protocol,
#         worker=mock_protocol_worker,
#         parallel=False,
#     )
#     pd.testing.assert_frame_equal(
#         result.variables,
#         pd.DataFrame(
#             {"x1": [1.0], "x2": [2.0]},
#             index=pd.Index(parameters["param1"]),
#         ),
#     )
#     pd.testing.assert_frame_equal(
#         result.fluxes,
#         pd.DataFrame(
#             {"v1": [0.1], "v2": [0.2]},
#             index=pd.Index(parameters["param1"]),
#         ),
#     )
