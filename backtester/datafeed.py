import numpy as np

from datetime import datetime
from abc import ABC, abstractmethod
from backtester.indicators import Indicator
from backtester.assets_data import AssetsData


class DataFeed(ABC):
    _idx: int
    _assets: AssetsData
    _symbols: list[str]
    _indicators: dict[str, Indicator]

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    @property
    @abstractmethod
    def idx(self) -> int:
        pass

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    @abstractmethod
    def current_datetime(self) -> datetime:
        pass

    @property
    @abstractmethod
    def data(self) -> AssetsData:
        pass

    @property
    @abstractmethod
    def indicators(self) -> dict[str, Indicator]:
        pass

    @abstractmethod
    def bid_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def ask_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def spot_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def add_indicator(self, indicator: Indicator, name: str) -> None:
        pass

    @abstractmethod
    def indicator(
        self, symbol: str, indicator_name: str, key: int | slice
    ) -> float | np.ndarray[np.float64]:
        pass

    @abstractmethod
    def price(
        self, symbol: str, price: str, key: int | slice
    ) -> float | np.ndarray[np.float64]:
        pass
