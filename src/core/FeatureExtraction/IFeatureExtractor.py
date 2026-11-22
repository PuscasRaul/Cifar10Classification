#!/usr/bin/env python3
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA

from src.core.models.Settings import settings


class IFeatureExtractor(ABC):
    def __init__(self) -> None:
        self._train_mean: NDArray | None = None
        self._std_dev: NDArray | None = None
        return None

    @abstractmethod
    def feature_extraction(self, x: NDArray) -> NDArray | None:
        return None

    def PCA(self, x: NDArray) -> NDArray:
        '''
        PCA algorithm implementation
        input:
            arr -> numpy NDArray to compute PCA for
            k -> number of principal axis components to keep
        output:
            y -> NDArray of principal axis components
        '''
        no_of_dims = round(np.sqrt(np.shape(x)[1]))
        arr = x.astype(float)
        if self._train_mean is None:
            mean = np.mean(arr, axis=0, keepdims=True)
            self._train_mean = mean
        np.subtract(arr, self._train_mean, out=arr)

        if self._std_dev is None:
            variance_squared = np.mean(arr ** 2, axis=0, keepdims=True)
            std_dev = np.sqrt(variance_squared)
            std_dev[std_dev == 0] = 1.0
            self._std_dev = std_dev

        np.divide(arr, self._std_dev, out=arr)

        transposed_array = np.transpose(arr)
        covariance = np.matmul(transposed_array, arr) / arr.shape[0]

        principal_values, principal_axis = LA.eigh(covariance)
        order = np.argsort(principal_values)[::-1]
        eigenvalues = principal_values[order]
        principal_axis = np.array(principal_axis)
        if settings.show_plots is True:
            self._plot_information_conservation(eigenvalues, no_of_dims)
        principal_axis = principal_axis[:, order]

        return np.array(principal_axis[:, :no_of_dims])

    def _plot_information_conservation(
            self,
            principal_values: NDArray,
            no_of_dims: int
    ):
        ratios = np.cumsum(principal_values) / np.sum(principal_values)
        factors = ratios[0:no_of_dims]
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
