#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA


class IFeatureExtractor(ABC):
    def __init__(self, arr: NDArray, k: int) -> None:
        self._arr = arr
        self._k = k
        return None

    @abstractmethod
    def feature_extraction(self) -> NDArray | None:
        return None

    def PCA(self) -> NDArray:
        '''
        PCA algorithm implementation
        input:
            arr -> numpy NDArray to compute PCA for
            k -> number of principal axis components to keep
        output:
            y -> NDArray of principal axis components
        '''
        self._arr = self._arr.astype(float)
        mean = np.mean(self._arr, axis=0, keepdims=True)
        np.subtract(self._arr, mean, out=self._arr)

        variance_squared = np.mean(self._arr ** 2, axis=0, keepdims=True)
        std_dev = np.sqrt(variance_squared)
        std_dev[std_dev == 0] = 1.0

        np.divide(self._arr, std_dev, out=self._arr)

        transposed_array = np.transpose(self._arr)
        covariance = np.matmul(transposed_array, self._arr) / self._arr.shape[0]

        principal_values, principal_axis = LA.eigh(covariance)
        order = np.argsort(principal_values)[::-1]
        principal_axis = principal_axis[:, order]

        return principal_axis[:, :self._k]
