#!/usr/bin/env python3
from typing import Any
import numpy as np
from numpy.typing import NDArray
import enum

from src.core.Classifier.IClassifier import IClassifier


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


# TODO: Speed this shit up
class DecisionTree(IClassifier):
    _dt: Node | None = None
    _heuristics: dict[StoppageHeuristic, Any] = {}

    def __init__(
            self,
            heuristics: dict[StoppageHeuristic, Any],
    ) -> None:
        self._heuristics = heuristics

    def fit(self, x, y):
        self._dt = self._split(x, y, 0)

    def predict(self, x):
        if self._dt is None:
            raise ValueError(
                "The classifier has not been fitted to a dataset yet"
            )
        return self._walk_tree(self._dt, x)

    def _walk_tree(self, node, x):
        if self._is_leaf(node):
            return node._value

        if x[node._feature_index] <= node._threshold:
            return self._walk_tree(x, node._left)
        return self._walk_tree(x, node._right)

    # TODO: Check for bug, it seems like gain seems content on remaining
    # 0.0
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
        if self._check_finalized_condition(current_depth, x, 0) is True:
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
            # Here, I think we can get away without computing information gain
            # For absolutely each and every one of the thresholds
            # Since we already filter by unique values, the only one that moves
            # To the right would be just one of them, right?
            # So with one small addition, after I look over the IG process
            # We should do it
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
        left_child = self._split(x_left, y_left, current_depth + 1)
        right_child = self._split(x_right, y_right, current_depth + 1)
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
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * np.log2(p)
        return h

    def _check_finalized_condition(
            self,
            current_depth,
            x,
            current_loss
    ):
        keys = self._heuristics.keys()
        for key in keys:
            match key:
                case StoppageHeuristic.DEPTH:
                    value = self._heuristics[key]
                    if self._apply_depth_criterion(
                            value,
                            current_depth
                    ) is False:
                        return False
                    continue
                case StoppageHeuristic.LEAF_SIZE:
                    value = self._heuristics[key]
                    if self._apply_leaf_criterion(x, value) is False:
                        return False
                    continue
                case StoppageHeuristic.LOSS_FUNCTION_DECREASE:
                    value = self._heuristics[key]
                    if self._apply_loss_criterion(current_loss, value) is False:
                        return False
                    continue
                case _:
                    return False
        return True

    def _apply_depth_criterion(self, tree_depth, current_depth) -> bool:
        if tree_depth <= current_depth:
            return True
        return False

    def _apply_leaf_criterion(self, x, leaf_size) -> bool:
        no_of_items = len(x)
        if no_of_items < leaf_size:
            return True
        return False

    def _apply_loss_criterion(self, current_loss, loss) -> bool:
        if current_loss <= loss:
            return True
        return False

    def _most_common_label(self, y) -> Any:
        values = np.unique(y, return_counts=True)
        return values[0]

    def _is_leaf(self, node: Node) -> bool:
        return (node._left is None and node._right is None)

    def _count_nodes(self, node) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node._left) + self._count_nodes(node._right)
