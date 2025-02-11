import pytest

from modelbase2.types import MockSurrogate


@pytest.fixture
def mock_surrogate() -> MockSurrogate:
    return MockSurrogate(
        args=["x"],
        stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
    )
