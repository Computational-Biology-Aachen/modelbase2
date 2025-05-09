import pytest

from mxlpy.types import MockSurrogate


@pytest.fixture
def mock_surrogate() -> MockSurrogate:
    return MockSurrogate(
        args=["x"],
        outputs=["v1"],
        stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
    )
