import asyncio
from collections.abc import Sequence
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Any

from src.core.Classifier.DecisionTree import (
    DecisionTree,
    StoppageHeuristic
)
from src.core.Classifier.IClassifier import IClassifier

logger = logging.getLogger()


class RandomForest(IClassifier):

    def __init__(
            self,
            dt: Sequence[DecisionTree],
            sample_ratio: float | None
    ) -> None:
        """
        @param: k: number of trees to work with
        """
        self._trees = dt
        self._ratio = sample_ratio

    def predict(self, x) -> Any:
        results = [tree.predict(x) for tree in self._trees]
        frequency: dict[int, int] = {}
        for i in range(len(results)):
            elem = frequency.get(results[i], -1)
            if elem == -1:
                frequency[results[i]] = 1
            else:
                frequency[results[i]] = frequency[results[i]] + 1

        max_freq = -1
        output_label = -1
        for k, v in frequency.items():
            if v > max_freq:
                output_label = k
                max_freq = v

        return output_label

    def fit(self, x: NDArray, y: NDArray) -> None:
        for tree in self._trees:
            self._fit_one_tree(x, y, tree)

    def _fit_one_tree(self, x, y, tree):
        x_subsample, y_subsample = self._subsample(x, y, self._ratio)
        tree.fit(x_subsample, y_subsample)

    def _subsample(
            self,
            x: NDArray,
            y: NDArray,
            ratio=1.0
    ) -> tuple[NDArray, NDArray]:
        dataset_size = np.shape(x)[0]
        n_sample = round(dataset_size * ratio)

        rng = np.random.default_rng()
        indexes = rng.integers(low=0, high=dataset_size, size=n_sample)
        x_sample = [x[index] for index in indexes]
        y_sample = [y[index] for index in indexes]

        return tuple((np.array(x_sample), np.array(y_sample)))
