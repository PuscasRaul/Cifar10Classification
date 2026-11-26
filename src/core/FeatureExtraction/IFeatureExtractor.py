#!/usr/bin/env python3
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA

from src.core.models.PCAResult import PCAResult
from src.core.models.Settings import settings


class IFeatureExtractor(ABC):
    def __init__(self, k) -> None:
        self._k = k
        # Necessary, for doing PCA on both HOG features and input pixels
        # Separately, to avoid the variance problem
        self._PCA_result: dict[str, PCAResult] = {}
        return None

    @abstractmethod
    def feature_extraction(
            self,
            x: NDArray,
    ) -> NDArray | None:
        return None

    def PCA(self, x: NDArray, key: str | None) -> NDArray:
        '''
        PCA algorithm implementation
        input:
            arr -> numpy NDArray to compute PCA for
            k -> number of principal axis components to keep
        output:
            y -> NDArray after Principal components * x
        '''
        arr = x.astype(float)
        result = self._PCA_result.get(key, None)
        if result:
            np.subtract(arr, result.mean, out=arr)
            np.divide(arr, result.std_dev, out=arr)
            return np.matmul(arr, result.primary_axis)

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
        principal_axis = principal_axis[:, order]
        principal_axis = principal_axis[:, :self._k]

        if settings.show_plots is True:
            self._plot_information_conservation(eigenvalues, self._k)

        self._PCA_result[key] = PCAResult(
            mean=mean,
            std_dev=std_dev,
            primary_axis=principal_axis
            )

        return np.matmul(x, principal_axis)

    def _plot_information_conservation(
            self,
            principal_values: NDArray,
            no_of_dims: int
    ):
        ratios = np.cumsum(principal_values) / np.sum(principal_values)
        plt.plot(
            range(1, len(ratios) + 1),
            ratios,
            'o-',
            linewidth=2,
            color='red'
        )
        plt.axvline(no_of_dims, linestyle='--', color='gray')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance explained')
        plt.show()
