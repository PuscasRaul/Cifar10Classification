#!/usr/bin/env python3
import pickle
import numpy as np
from typing import Any
from src.core.Dataset import Dataset


class CifarDataset(Dataset):
    _base_path: str | None = None
    _image_size: int = 32
    _pixels_per_clr: int = 1024

    def __init__(self, file_path: str) -> None:
        self._base_path = file_path
        _database_dict = self._unpickle(file_path)
        self._x = self._get_images(_database_dict)
        self._y = self._get_labels(_database_dict)

    def get_X(self) -> list[Any]:
        return self._x

    def get_Y(self) -> list[Any]:
        return self._y

    def _unpickle(self, file: str) -> dict[Any, Any]:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _get_labels(self, dataset) -> list[int]:
        return dataset[b'labels']

    def _get_images(self, dataset):
        rows = dataset[b'data']
        result = []
        for row in rows:
            result.append(self._compose_image(row))
        return result

    def _compose_image(self, row) -> np.typing.NDArray[np.uint8]:
        image = np.empty(
            (self._image_size, self._image_size, 3),
            dtype=np.uint8
        )
        red_values = row[:self._pixels_per_clr]
        green_values = row[self._pixels_per_clr: self._pixels_per_clr * 2]
        blue_values = row[-self._pixels_per_clr:]

        for i in range(0, self._pixels_per_clr, self._image_size):
            red_pixels = red_values[i:i+self._image_size]
            green_pixels = green_values[i:i+self._image_size]
            blue_pixels = blue_values[i:i+self._image_size]
            row_idx = i // self._image_size
            image[row_idx] = self._compose_pixel_row(
                red_pixels,
                green_pixels,
                blue_pixels)
        return image

    def _compose_pixel_row(
            self,
            red,
            green,
            blue
    ) -> np.typing.NDArray[np.uint8]:
        pixel_row = np.empty((self._image_size, 3), dtype=np.uint8)
        for i in range(0, self._image_size):
            pixel_row[i] = self._zip_pixels(red[i], blue[i], green[i])
        return pixel_row

    def _zip_pixels(
            self,
            red: int,
            green: int,
            blue: int
    ) -> np.typing.NDArray:
        return np.array((red, green, blue), dtype=np.uint8)
