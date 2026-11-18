#!/usr/bin/env python3
from numpy.typing import NDArray

from src.core.IFeatureExtractorInterface import IFeatureExtractorInterface


class PCAFeatureExtractor(IFeatureExtractorInterface):

    def __init__(self, arr: NDArray, k: int) -> None:
        """
        arr: @input: 2 dimensional NDArray of (x, y)
        k: @input: final x dimension after PCA
        """
        IFeatureExtractorInterface.__init__(self, arr, k)
        return None

    def feature_extraction(self) -> NDArray:
        return self.PCA()
