"""Example models.

This package provides functions for retrieving various predefined metabolic models.
It includes models for example systems, linear chains, and specific metabolic pathways.

"""

from __future__ import annotations

from .example1 import get_example1
from .linear_chain import get_lin_chain_two_circles, get_linear_chain_2v
from .poolman2000 import get_model as get_poolman2000
from .upper_glycolysis import get_model as get_upper_glycolysis

__all__ = [
    "get_example1",
    "get_lin_chain_two_circles",
    "get_linear_chain_2v",
    "get_poolman2000",
    "get_upper_glycolysis",
]