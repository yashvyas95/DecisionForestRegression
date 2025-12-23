"""
Core module initialization for Decision Forest Regression.

This module provides the main classes and functions for the decision forest
regression algorithm implementation.
"""

from .core.decision_tree import DecisionTree
from .core.forest import DecisionForest
from .core.node import Node
from .core.splitters import BestSplitter, RandomSplitter

__version__ = "2.0.0"
__author__ = "Yash Vyas"
__email__ = "yash.vyas@example.com"

__all__ = [
    "DecisionTree",
    "DecisionForest", 
    "Node",
    "BestSplitter",
    "RandomSplitter",
]