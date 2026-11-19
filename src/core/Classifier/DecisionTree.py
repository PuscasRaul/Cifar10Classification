#!/usr/bin/env python3
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray
import enum

from src.core.Classifier.IClassifier import IClassifier

logger = logging.getLogger(__name__)


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


@dataclass
class TreeStats:
    n_train: int = 0
    fit_time: float = 0.0
    leaf_depths: list[int] = field(default_factory=list)
    leaf_sizes: list[int] = field(default_factory=list)
    importances: NDArray = field(default_factory=lambda: np.zeros(0))


# TODO: Speed this shit up
class DecisionTree(IClassifier):
    _dt: Node | None = None
    _heuristics: dict[StoppageHeuristic, Any] = {}

    def __init__(
            self,
            heuristics: dict[StoppageHeuristic, Any],
    ) -> None:
        self._heuristics = heuristics
        self._stats = TreeStats()

    @property
    def stats(self) -> TreeStats:
        return self._stats

    @property
    def feature_importances(self) -> NDArray:
        return self._stats.importances

    def fit(self, x, y):
        start = time.perf_counter()
        self._stats = TreeStats(
            n_train=len(y),
            importances=np.zeros(x.shape[1]),
        )
        self._dt = self._split(x, y, 0)

        total = self._stats.importances.sum()
        if total > 0:
            self._stats.importances /= total  # normalise to sum to 1
        self._stats.fit_time = time.perf_counter() - start
        self._log_fit_summary()

    def predict(self, x):
        """
        @input x: THIS IS JUST ONE vector, NOT THE WHOLE DB
        """
        if self._dt is None:
            raise ValueError(
                "The classifier has not been fitted to a dataset yet"
            )
        return self._walk_tree(self._dt, x)

    def _walk_tree(self, node, x):
        if self._is_leaf(node):
            return node._value

        if x[node._feature_index] <= node._threshold:
            return self._walk_tree(node._left, x)
        return self._walk_tree(node._right, x)

    def _split(self, x: NDArray, y: NDArray, current_depth: int) -> Node:
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
            return self._make_leaf(y, current_depth, "stop criterion")
        if len(np.unique(y)) == 1:
            return self._make_leaf(y, current_depth, "pure")

        best_gain = 0.0
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

        if best_feature_idx is None:
            return self._make_leaf(y, current_depth, "no gain")

        self._stats.importances[best_feature_idx] += (
            len(y) / self._stats.n_train * float(best_gain)
        )
        logger.debug(
            "SPLIT depth=%d feature=%d threshold=%.4f gain=%.4f left=%d right=%d",
            current_depth, best_feature_idx, best_threshold, best_gain,
            best_left_mask.sum(), best_right_mask.sum(),
        )

        x_left, y_left = x[best_left_mask], y[best_left_mask]
        x_right, y_right = x[best_right_mask], y[best_right_mask]
        left_child = self._split(x_left, y_left, current_depth + 1)
        right_child = self._split(x_right, y_right, current_depth + 1)
        return Node(feature_index=best_feature_idx,
                    threshold=best_threshold,
                    left=left_child,
                    right=right_child)

    def _make_leaf(self, y: NDArray, depth: int, reason: str) -> Node:
        label = self._most_common_label(y)
        self._stats.leaf_depths.append(depth)
        self._stats.leaf_sizes.append(len(y))
        if logger.isEnabledFor(logging.DEBUG):
            _, counts = np.unique(y, return_counts=True)
            logger.debug(
                "LEAF  depth=%d n=%d label=%s purity=%.2f reason=%s",
                depth, len(y), label, counts.max() / len(y), reason,
            )
        return Node(value=label)

    def _log_fit_summary(self) -> None:
        s = self._stats
        depths = np.array(s.leaf_depths)
        sizes = np.array(s.leaf_sizes)
        logger.info(
            "Fit in %.3fs | nodes=%d leaves=%d | max depth=%d, mean leaf depth=%.2f",
            s.fit_time, self._count_nodes(self._dt), len(depths),
            depths.max(), depths.mean(),
        )
        logger.info(
            "Leaf size: min=%d mean=%.1f max=%d",
            sizes.min(), sizes.mean(), sizes.max(),
        )
        top = np.argsort(s.importances)[::-1][:5]
        logger.info(
            "Top features: %s",
            ", ".join(f"#{i} ({s.importances[i]:.3f})"
                      for i in top if s.importances[i] > 0),
        )

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
            x: NDArray,
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
        return x.shape[0] < leaf_size

    def _apply_loss_criterion(self, current_loss, loss) -> bool:
        if current_loss <= loss:
            return True
        return False

    def _most_common_label(self, y) -> Any:
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _is_leaf(self, node: Node) -> bool:
        return (node._left is None and node._right is None)

    def _count_nodes(self, node) -> int:
        if node is None:
            return 0
        return (
            1
            + self._count_nodes(node._left)
            + self._count_nodes(node._right)
        )
