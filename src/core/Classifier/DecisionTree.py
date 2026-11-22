#!/usr/bin/env python3
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.Classifier.IClassifier import IClassifier

logger = logging.getLogger(__name__)


class StoppageHeuristic(enum.Enum):
    DEPTH = 0
    LEAF_SIZE = 1
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


class DecisionTree(IClassifier):

    def __init__(
            self,
            heuristics: dict[StoppageHeuristic, Any],
            k_sample_dimensions: int | None = None
    ) -> None:
        self._dt: Node | None = None
        self._heuristics = heuristics
        self._stats = TreeStats()
        self._k_sample_dim = k_sample_dimensions
        self._rng = np.random.default_rng()

    @property
    def stats(self) -> TreeStats:
        return self._stats

    @property
    def feature_importances(self) -> NDArray:
        return self._stats.importances

    def fit(self, x: NDArray, y: NDArray):
        start = time.perf_counter()
        self._stats = TreeStats(
            n_train=len(y),
            importances=np.zeros(x.shape[1]),
        )
        self._dt = self._split(np.asarray(x), np.asarray(y), 0)

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
        while not self._is_leaf(node):
            if x[node._feature_index] <= node._threshold:
                node = node._left
            else:
                node = node._right
        return node._value

    def _split(self, x: NDArray, y: NDArray, current_depth: int) -> Node:
        if self._check_finalized_condition(current_depth, x.shape[0]):
            return self._make_leaf(y, current_depth, "stop criterion")

        classes, y_idx = np.unique(y, return_inverse=True)
        if len(classes) == 1:
            return self._make_leaf(y, current_depth, "pure")

        best_gain, best_feature_idx, best_threshold = self._best_split(
            x, y_idx, len(classes)
        )

        if best_feature_idx is None:
            return self._make_leaf(y, current_depth, "no gain")

        # TODO: move this out of here, why are we applying this here?
        min_gain = self._heuristics.get(
            StoppageHeuristic.LOSS_FUNCTION_DECREASE
        )
        if min_gain is not None and best_gain <= min_gain:
            return self._make_leaf(y, current_depth, "gain too small")

        left_mask = x[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask

        self._stats.importances[best_feature_idx] += (
            len(y) / self._stats.n_train * best_gain
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SPLIT depth=%d feature=%d threshold=%.4f gain=%.4f "
                "left=%d right=%d",
                current_depth, best_feature_idx, best_threshold, best_gain,
                left_mask.sum(), right_mask.sum(),
            )

        left_child = self._split(x[left_mask], y[left_mask], current_depth + 1)
        right_child = self._split(
            x[right_mask], y[right_mask], current_depth + 1
        )
        return Node(feature_index=best_feature_idx,
                    threshold=best_threshold,
                    left=left_child,
                    right=right_child)

    def _best_split(self, x, y_idx, n_classes):
        n, n_features = x.shape

        # Required for Random Forest
        # set k_sample_dim = sqrt(n_features),
        # don't ask it was in the Princeton course
        # i forgot about the reasoning

        if self._k_sample_dim is not None:
            candidate_features = self._rng.choice(
                    n_features,
                    size=self._k_sample_dim,
                    replace=False
            )
            candidate_features = np.ravel(candidate_features)
        else:
            candidate_features = range(n_features)
        all_counts = np.bincount(y_idx, minlength=n_classes).tolist()
        parent_entropy = self._entropy(all_counts, n)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in candidate_features:
            order = np.argsort(x[:, feature])
            values = x[order, feature].tolist()
            labels = y_idx[order].tolist()

            left = [0] * n_classes
            right = list(all_counts)

            for i in range(n - 1):
                left[labels[i]] += 1
                right[labels[i]] -= 1

                if values[i] == values[i + 1]:
                    continue

                n_left = i + 1
                n_right = n - n_left
                child_entropy = (
                    n_left * self._entropy(left, n_left)
                    + n_right * self._entropy(right, n_right)
                ) / n
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = values[i]

        return best_gain, best_feature, best_threshold

    def _entropy(self, counts: NDArray, no_of_labels: int) -> np.float32:
        """
        Compute entropy function.
        @input counts: array, where each element represents the frequency of a
                       label
        @input no_of_labels: total number of labels
        """
        probs = [count / no_of_labels for count in counts]

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * np.log2(p)
        return h

    def _check_finalized_condition(self, current_depth: int, n_samples: int):
        for key, value in self._heuristics.items():
            match key:
                case StoppageHeuristic.DEPTH:
                    if current_depth >= value:
                        return True
                case StoppageHeuristic.LEAF_SIZE:
                    if n_samples < value:
                        return True
        return False

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
            "Fit in %.3fs | nodes=%d leaves=%d | max depth=%d, "
            "mean leaf depth=%.2f",
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
