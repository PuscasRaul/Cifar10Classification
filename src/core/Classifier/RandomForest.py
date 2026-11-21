import asyncio
from collections.abc import Sequence
import logging
import numpy as np
from numpy.typing import NDArray
from random import randrange
from typing import Any

from src.core.Classifier.DecisionTree import (
    DecisionTree,
    StoppageHeuristic
)
from src.core.Classifier.IClassifier import IClassifier

logger = logging.getLogger()


class RandomForest(IClassifier):
    # TODO: Check how to interact with the stoppage criteria
    #       Theoretically I know we should just grow the trees completely
    #       But I might be wrong

    # TODO: Check if mixing Sequence and numpy arrays in a codebase is
    #       considered a good practice. I do not see why not

    _trees: Sequence[DecisionTree] = []
    _ratio: float | None = None

    def __init__(
            self,
            k: int,
            heuristic: dict[StoppageHeuristic, Any],
            sample_ratio: float | None
    ) -> None:
        """
        @param: k: number of trees to work with
        """
        if k == 0:
            raise ValueError("Number of trees must not be 0")
        for i in range(0, k):
            self._trees.append(DecisionTree, heuristic)
        self._ratio = sample_ratio

    async def predict(self, x) -> Any:
        # TODO: check, but this should probably do it for the moment
        #       more importantly, make the tree itself fucking faster
        results = await asyncio.gather(
            *(tree.predict(x) for tree in self._trees)
        )
        frequency: dict[int, int] = {}
        for i in range(results):
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

    async def fit(self, x: NDArray, y: NDArray) -> None:
        no_of_trees = len(self._trees)
        for i in range(no_of_trees):
            x_subsample, y_subsample = self._subsample(x, self._ratio)
            self._trees[i].fit(x_subsample, y_subsample)

    def _subsample(
            self,
            x: NDArray,
            y: NDArray,
            ratio=1.0
    ) -> tuple[NDArray, NDArray]:
        # TODO: Check in debugger
        dataset_size = x.size()
        x_sample = []
        y_sample = []
        n_sample = round(dataset_size / ratio)

        while sample.size() < n_sample:
            index = randrange(dataset_size)
            x_sample.append(x.at(index))
            y_sample.append(y.at(index))

        return tuple((np.array(x_sample), np.array(y_sample))
