#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA


class FeatureExtractorInterface(ABC):
    def __init__(self) -> None:
        return None

    @abstractmethod
    def feature_extraction(self) -> None:
        return None

    def PCA(self, arr: NDArray, k: int) -> NDArray:
        '''
        PCA algorithm implementation
        input:
            arr -> numpy NDArray to compute PCA for
            k -> number of principal axis components to keep
        output:
            y -> NDArray of principal axis components
        '''
        arr = arr.astype(float)
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
        principal_axis = principal_axis[:, order]

        return principal_axis[:, :k]
