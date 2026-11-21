import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.FeatureExtraction.IFeatureExtractor import IFeatureExtractor


class HandcraftedFeatureExtractor(IFeatureExtractor):

    def __init__(self, arr: NDArray, k: int) -> None:
        """
        @input: arr -> ndarray of x vectors
        @input: k -> int to describe the amount of dimensions after PCA
        """
        IFeatureExtractor.__init__(self, arr, k)

    def feature_extraction(self) -> NDArray | None:
        self._arr = self._hog_features()
        primary_components = self.PCA()
        return np.matmul(self._arr, primary_components)

    def _hog_features(self) -> NDArray:
        """
        Extract hog features from the input array
        """
        features = [self._hog_descriptor(img) for img in self._arr]
        return np.array(features)

    def _hog_descriptor(self, x: NDArray) -> NDArray:
        cell_size = 8
        h, w = x.shape
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        x = x[:n_cells_y * 8, :n_cells_x * 8].astype(np.float32) / 255.0

        patches = x.reshape(
            n_cells_y, 8,
            n_cells_x, 8
        ).transpose(0, 2, 1, 3)

        histograms = np.zeros((n_cells_y, n_cells_x, 9))

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                patch = patches[i, j]
                gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
                mag, angle = cv2.cartToPolar(
                    gx, gy, angleInDegrees=True
                )
                angle %= 180
                histograms[i, j] = self._histogram_of_gradients(
                    mag, angle
                )

        descriptor = []
        for i in range(n_cells_y - 1):
            for j in range(n_cells_x - 1):
                block = histograms[i:i+2, j:j+2, :]
                block = block.flatten()
                eps = 1e-5
                block /= np.sqrt(np.sum(block ** 2) + eps ** 2)
                block = np.clip(block, 0, 0.2)
                block /= np.sqrt(np.sum(block ** 2) + eps ** 2)
                descriptor.append(block)

        return np.concatenate(descriptor)

    def _histogram_of_gradients(
        self,
        mag: NDArray,
        angle: NDArray
    ) -> NDArray:
        """
        Compute orientation histogram for an 8x8 cell.
        """

        bins = np.zeros(9, dtype=np.float32)
        bin_width = 180 / 9

        bin_indices = angle / bin_width

        lower_bins = np.floor(bin_indices).astype(int) % 9
        upper_bins = (lower_bins + 1) % 9

        weights = bin_indices - np.floor(bin_indices)

        np.add.at(bins, lower_bins, mag * (1 - weights))
        np.add.at(bins, upper_bins, mag * weights)

        return bins
