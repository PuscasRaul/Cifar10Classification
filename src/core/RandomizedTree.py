#!/usr/bin/env python3
from typing import Any
import numpy as np
from numpy.typing import NDArray
import enum

from src.core.Classifier import Classifier


class StoppageHeuristic(enum.Enum):
    DEPTH = 0,
    LEAF_SIZE = 1,
    LOSS_FUNCTION_DECREASE = 3


class Node():
    def __init__(
            self,
            feature_index=None,
            threshold=None,
            left=None,
            right=None,
            value=None
    ) -> None:
        self._feature_index = feature_index
        self._threshold = threshold
        self._left = left
        self._right = right
        self._value = value


class RandomizedTree(Classifier):
    _dt: Node | None = None
    _heuristics: dict[StoppageHeuristic, Any] = {}

    def __init__(
            self,
            heuristics: dict[StoppageHeuristic, Any],
            x: NDArray,
            y: NDArray
    ) -> None:
        self._heuristics = heuristics

    def fit(self, x, y):

    def predict(self, x):

    def _walk_tree(self, node, x):

    def _split(self, x: NDArray, y: NDArray, current_depth: int) -> Node | None:
        '''
        The main idea is given the original set S
        We need to find the best 2 splits for the data, in order to minimize
        the entropy/ maximize the information gain

        By splitting the data, we mean taking one attribute of the input vector
        And verifying it for each individual value, calculating information gain
        and checking which one is maximum.

        After computing the maximum, we run this algorithm again with
        _split(S_Left)
        _split(S_Right)

        And somehow, at the end
        Make the big fucking tree
        '''

    def _information_gain(
            self,
            y_parent: NDArray,
            y_left: NDArray,
            y_right: NDArray) -> np.float32:

    def _entropy(self, labels: NDArray) -> np.float32:
