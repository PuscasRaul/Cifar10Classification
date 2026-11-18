import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.Preprocessing.IPreprocessor import IPreprocessor


class PreprocessorV1(IPreprocessor):

    def __init__(self, size: int) -> None:
        """
        @input: size -> size to be resized to
        """
        self._size = size
        return None

    def process_vectors(self, x: NDArray) -> NDArray:
        """
        @input: x NDArray of vectors to be processed
        @output: NDArray of same shape as input, where each corresponding vector
                 represents the same element, but transformed
        """
        x = [cv2.resize(vec, (self._size, self._size)) for vec in x]
        x = [cv2.cvtColor(vec, cv2.COLOR_BGR2GRAY, vec) for vec in x]
        x = [cv2.GaussianBlur(vec, (3, 3), 0) for vec in x]

        return np.array(x)
