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
        self._x: np.typing.NDArray[np.uint8] = (
            self._get_images(_database_dict)
            .reshape(-1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE)
            .transpose(0, 2, 3, 1)
        )
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
        return rows

