from collections.abc import Sequence
import logging
import numpy as np
import math
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import Any

from src.core.Classifier.DecisionTree import (
    DecisionTree,
)
from src.core.Classifier.IClassifier import IClassifier

logger = logging.getLogger()
NO_OF_WORKERS = 10


def _subsample(
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


def _fit_chunk(
        trees: Sequence[DecisionTree],
        x: NDArray,
        y: NDArray,
        ratio
):
    fitted_trees: list[DecisionTree] = []
    for tree in trees:
        x_subsample, y_subsample = _subsample(x, y, ratio)
        tree.fit(x_subsample, y_subsample)
        fitted_trees.append(tree)
    return fitted_trees


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
        pool = Pool(processes=NO_OF_WORKERS)
        num_workers = min(NO_OF_WORKERS, len(self._trees))
        chunk_size = math.ceil(len(self._trees) / num_workers)
        ratio = self._ratio

        tasks = [
            (self._trees[i: i + chunk_size], x, y, ratio)
            for i in range(0, len(self._trees), chunk_size)
        ]

        fitted_chunks = pool.starmap(_fit_chunk, tasks)
        self._trees = [tree for chunk in fitted_chunks for tree in chunk]
        pool.close()
        pool.join()
