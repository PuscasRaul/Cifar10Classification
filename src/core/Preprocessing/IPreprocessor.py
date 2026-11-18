from numpy.typing import NDArray
from abc import ABC, abstractmethod


class IPreprocessor(ABC):

    def __init__(self) -> None:
        return None

    @abstractmethod
    def process_vectors(self, x: NDArray) -> NDArray:
        """
        @input: x NDArray of vectors to be processed
        @output: NDArray of same shape as input, where each corresponding vector
                 represents the same element, but transformed
        """
        return None
