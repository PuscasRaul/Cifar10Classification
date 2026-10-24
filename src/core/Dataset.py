#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Any


class Dataset(ABC):
    _x: list[Any] = []
    _y: list[Any] = []

    def __init__(self) -> None:
        return None

    @abstractmethod
    def get_X(self) -> list[Any]:
        ''' Return a list of vectors that represent the training data'''

    @abstractmethod
    def get_Y(self) -> list[Any]:
        ''' Return a list of labels for the training vectors'''
