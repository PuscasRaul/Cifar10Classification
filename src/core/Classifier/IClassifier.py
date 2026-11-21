#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Any


class IClassifier(ABC):
    def __init__() -> None:
        return None

    @abstractmethod
    async def predict(self, x) -> Any:
        return None

    '''
    Maybe have the metrics defined here
    '''
    @abstractmethod
    async def fit(self, x, y) -> None:
        return None
