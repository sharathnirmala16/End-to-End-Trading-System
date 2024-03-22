from backtester.indicators import Indicator
from abc import ABC, abstractmethod
from datetime import datetime


class DataFeed(ABC):
    _symbols: list[str]
    _indicators: dict[str, Indicator]

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    @abstractmethod
    def current_datetime(self) -> datetime:
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
