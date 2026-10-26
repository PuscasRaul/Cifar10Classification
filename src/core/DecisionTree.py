#!/usr/bin/env python3
from typing import Any
import numpy as np
from numpy.typing import NDArray
import enum


class StoppageHeuristic(enum.Enum):
    DEPTH = 0,
    LEAF_SIZE = 1,
    NO_OF_NODES = 2,
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


class DecisionTree():
    _dt: Node | None = None
    _heuristics: dict[StoppageHeuristic, Any] = {}

    def __init__(
            self,
            heuristics: dict[StoppageHeuristic, Any],
            x: NDArray,
            y: NDArray
    ) -> None:
        self._heuristics = heuristics
        self._dt = self._split(x, y)

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
        if self._check_finalized_condition(current_depth, y) is True:
            return Node(value=self._most_common_label(y))

        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        n_dimensions = x.shape[1]

        for i in range(n_dimensions):
            feature_column = x[:, i]
            unique_values = np.unique(feature_column)
            for threshold in unique_values:
                left_mask = feature_column <= threshold
                right_mask = feature_column > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = i
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        if best_gain == -1:
            return Node(value=self._most_common_label(y))

        x_left, y_left = x[best_left_mask], y[best_left_mask]
        x_right, y_right = x[best_right_mask], y[best_right_mask]
        left_child = self._split(x_left, y_left)
        right_child = self._split(x_right, y_right)
        return Node(feature_index=best_feature_idx,
                    threshold=best_threshold,
                    left=left_child,
                    right=right_child)

    def _information_gain(
            self,
            y_parent: NDArray,
            y_left: NDArray,
            y_right: NDArray) -> np.float32:

        n_total = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        h_s = self._entropy(y_parent)
        h_l = self._entropy(y_left)
        h_r = self._entropy(y_right)

        weighted_left = h_l * (n_left / n_total)
        weighted_right = h_r * (n_right / n_total)

        return np.float32(h_s - (weighted_left + weighted_right))

    def _entropy(self, labels: NDArray) -> np.float32:
        if len(labels == 0):
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * np.log2(p)
        return h

    def _check_finalized_condition(self, current_depth, y):
        keys = self._heuristics.keys()
        for key in keys:
            match key:
                case StoppageHeuristic.DEPTH:
                    value = self._heuristics[key]
                    if self._apply_depth_criterion(value) is False:
                        return False
                    continue
                case StoppageHeuristic.NO_OF_NODES:
                    value = self._heuristics[key]
                    if self._apply_nodes_criterion(value) is False:
                        return False
                    continue
                case StoppageHeuristic.LEAF_SIZE:
                    value = self._heuristics[key]
                    if self._apply_leaf_criterion(value) is False:
                        return False
                    continue
                case StoppageHeuristic.LOSS_FUNCTION_DECREASE:
                    value = self._heuristics[key]
                    if self._apply_loss_criterion(value) is False:
                        return False
                    continue
                case _:
                    return False
        return True

    def _apply_depth_criterion(self, tree_depth) -> bool:
        return False

    def _apply_nodes_criterion(self, no_of_nodes) -> bool:
        return False

    def _apply_leaf_criterion(self, leaf_size) -> bool:
        return False

    def _apply_loss_criterion(self, loss) -> bool:
        return False

    def _most_common_label(y) -> Any:
        values = np.unique(y, return_counts=True)
        return values[0]
