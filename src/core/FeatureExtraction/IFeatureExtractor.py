#!/usr/bin/env python3
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA

from src.core.models.Settings import settings


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
        arr = self._arr.astype(float)
        mean = np.mean(arr, axis=0, keepdims=True)
        np.subtract(arr, mean, out=arr)

        variance_squared = np.mean(arr ** 2, axis=0, keepdims=True)
        std_dev = np.sqrt(variance_squared)
        std_dev[std_dev == 0] = 1.0

        np.divide(arr, std_dev, out=arr)

        transposed_array = np.transpose(arr)
        covariance = np.matmul(transposed_array, arr) / arr.shape[0]

        principal_values, principal_axis = LA.eigh(covariance)
        order = np.argsort(principal_values)[::-1]
        eigenvalues = principal_values[order]
        principal_axis = np.array(principal_axis)
        if settings.show_plots is True:
            self._plot_information_conservation(eigenvalues)
        principal_axis = principal_axis[:, order]

        return np.array(principal_axis[:, :self._k])

    def _plot_information_conservation(self, principal_values: NDArray):
        ratios = np.cumsum(principal_values) / np.sum(principal_values)
        factors = ratios[0:self._k]
        plt.plot(
            range(1, len(factors) + 1),
            factors,
            'o-',
            linewidth=2,
            color='red'
        )
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance explained')
        plt.show()
