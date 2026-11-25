import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.FeatureExtraction.IFeatureExtractor import IFeatureExtractor


class HandcraftedFeatureExtractor(IFeatureExtractor):

    def __init__(self, k: int) -> None:
        IFeatureExtractor.__init__(self, k)

    def feature_extraction(self, x: NDArray) -> NDArray | None:
        """
        Computes hog features and extracts the first sqrt(feature_dim) PC
        @input x: vector of elements, where each element is flattened
        """
        K, w, _, _ = np.shape(x)
        red_values = np.reshape(x[:, :, :, 0], (K, w ** 2))
        green_values = np.reshape(x[:, :, :, 1], (K, w ** 2))
        blue_values = np.reshape(x[:, :, :, 2], (K, w ** 2))

        features = self._hog_features(x)  # K x no_of_hog_features
        pc_features = self.PCA(features, key="HOG")
        red_features = self.PCA(red_values, key="RED")
        green_features = self.PCA(green_values, key="GREEN")
        blue_features = self.PCA(blue_values, key="BLUE")

        return np.concatenate(
            [
                pc_features,
                red_features,
                green_features,
                blue_features
            ],
            axis=1
        )

    # TODO: this does not particularly pair well with HOG, since that
    #       requires greyscale images
    def _first_color_moment(self, x: NDArray) -> NDArray:
        return np.mean(x, axis=(1, 2))

    def _second_color_moment(
            self,
            x: NDArray,
            first_color_moments: NDArray
    ) -> NDArray:
        """
        So the first_color_moments are K x C
            K-no of samples
            C - mean per channel

        So for variance, we need to also somehow collapse
        those N x N internal axes to compute this
        """

        means = first_color_moments[:, None, None, :]
        squared_variance = (x - means) ** 2
        return np.sqrt(np.mean(squared_variance, axis=(1, 2)))

    def _third_color_moment(
            self,
            x: NDArray,
            first_color_moments: NDArray,
            second_color_moments: NDArray
    ) -> NDArray:
        means = first_color_moments[:, None, None, :]
        cubed_variance = (x - means) ** 3
        cbrt_variance = np.cbrt(np.mean(cubed_variance, axis=(1, 2)))
        return np.divide(cbrt_variance, second_color_moments)

    def _hog_features(self, x: NDArray) -> NDArray:
        """
        Extract hog features from the input array
        """
        features = [self._hog_descriptor(vec) for vec in x]
        return np.array(features)

    def _hog_descriptor(self, x: NDArray) -> NDArray:
        if x.ndim == 2:
            gray = x
        else:
            gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        h, w = np.shape(gray)
        cell_size = 8
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        gray = gray[:n_cells_y * 8, :n_cells_x * 8].astype(np.float32) / 255.0

        patches = gray.reshape(
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
