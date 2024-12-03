from collections.abc import Callable

import numpy as np
import pandas as pd

from example_models import get_linear_chain_2v
from modelbase2 import fit
from modelbase2.fit import ResidualFn
from modelbase2.fns import constant
from modelbase2.model import Model
from modelbase2.types import Array, ArrayLike, IntegratorProtocol, unwrap


def mock_minimize_fn(
    residual_fn: ResidualFn,  # noqa: ARG001
    p0: dict[str, float],
) -> dict[str, float]:
    return p0


def mock_residual_fn_filled_in(
    par_values: Array,  # noqa: ARG001
) -> float:
    return 0.0


def mock_ss_residual_fn(
    par_values: Array,  # noqa: ARG001
    par_names: list[str],  # noqa: ARG001
    data: pd.Series,  # noqa: ARG001
    model: Model,  # noqa: ARG001
    y0: dict[str, float],  # noqa: ARG001
    integrator: type[IntegratorProtocol],  # noqa: ARG001
) -> float:
    return 0.0


def mock_ts_residual_fn(
    par_values: Array,  # noqa: ARG001
    par_names: list[str],  # noqa: ARG001
    data: pd.DataFrame,  # noqa: ARG001
    model: Model,  # noqa: ARG001
    y0: dict[str, float],  # noqa: ARG001
    integrator: type[IntegratorProtocol],  # noqa: ARG001
) -> float:
    return 0.0


class MockIntegrator:
    def __init__(
        self,
        rhs: Callable,  # noqa: ARG002
        y0: ArrayLike,
    ) -> None:
        self.y0 = y0

    def reset(self) -> None:
        return

    def integrate(
        self,
        *,
        t_end: float | None = None,  # noqa: ARG002
        steps: int | None = None,  # noqa: ARG002
        time_points: ArrayLike | None = None,  # noqa: ARG002
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        t = np.array([0.0])
        y = np.ones((1, len(self.y0)))
        return t, y

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,  # noqa: ARG002
        rel_norm: bool,  # noqa: ARG002
    ) -> tuple[float | None, ArrayLike | None]:
        t = 0.0
        y = np.ones(len(self.y0))
        return t, y


def test_default_minimize_fn() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_fit = fit._default_minimize_fn(
        mock_residual_fn_filled_in,
        p_true,
    )
    assert np.allclose(pd.Series(p_fit), pd.Series(p_true), rtol=0.1)


def test_steady_state_residual() -> None:
    model = (
        Model()
        .add_parameters({"k1": 1.0})
        .add_variables({"x1": 1.0})
        .add_reaction("v1", constant, stoichiometry={"x1": 1.0}, args=["k1"])
    )

    residual = fit._steady_state_residual(
        par_values=np.array([1.0]),
        par_names=["k1"],
        data=pd.Series({"x1": 1.0, "v1": 1.0}),
        model=model,
        integrator=MockIntegrator,
        y0={"x1": 1.0},
    )
    assert residual == 0.0


def test_time0_series_residual() -> None:
    model = (
        Model()
        .add_parameters({"k1": 1.0})
        .add_variables({"x1": 1.0})
        .add_reaction("v1", constant, stoichiometry={"x1": 1.0}, args=["k1"])
    )

    residual = fit._time_course_residual(
        par_values=np.array([1.0]),
        par_names=["k1"],
        data=pd.DataFrame({0.0: {"x1": 1.0, "v1": 1.0}}).T,
        model=model,
        integrator=MockIntegrator,
        y0={"x1": 1.0},
    )
    assert residual == 0.0


def test_fit_steady_state() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    data = pd.Series()
    p_fit = fit.steady_state(
        model=Model(),
        p0=p_true,
        data=data,
        minimize_fn=mock_minimize_fn,
        residual_fn=mock_ss_residual_fn,
    )
    assert np.allclose(pd.Series(p_fit), pd.Series(p_true), rtol=0.1)


def tets_fit_time_course() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    data = pd.DataFrame()
    p_fit = fit.time_course(
        model=Model(),
        p0=p_true,
        data=data,
        minimize_fn=mock_minimize_fn,
        residual_fn=mock_ts_residual_fn,
    )
    assert np.allclose(pd.Series(p_fit), pd.Series(p_true), rtol=0.1)


if __name__ == "__main__":
    from modelbase2 import Simulator

    model_fn = get_linear_chain_2v
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_init = {"k1": 1.038, "k2": 1.87, "k3": 1.093}
    res = unwrap(
        Simulator(model_fn())
        .update_parameters(p_true)
        .simulate(time_points=np.linspace(0, 1, 11))
        .get_results()
    )

    p_fit = fit.steady_state(
        model_fn(),
        p0=p_init,
        data=res.iloc[-1],
    )
    assert np.allclose(pd.Series(p_fit), pd.Series(p_true), rtol=0.1)

    p_fit = fit.time_course(
        model_fn(),
        p0=p_init,
        data=res,
    )
    assert np.allclose(pd.Series(p_fit), pd.Series(p_true), rtol=0.1)
