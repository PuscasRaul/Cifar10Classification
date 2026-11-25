from numpy.typing import NDArray


class PCAResult():
    def __init__(
            self,
            mean: NDArray,
            std_dev: NDArray,
            primary_axis: NDArray
    ):
        self._mean = mean
        self._std_dev = std_dev
        self._primary_axis = primary_axis

    @property
    def primary_axis(self) -> NDArray:
        return self._primary_axis

    @property
    def std_dev(self) -> NDArray:
        return self._std_dev

    @property
    def mean(self) -> NDArray:
        return self._mean
