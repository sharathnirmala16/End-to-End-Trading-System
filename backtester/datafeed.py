from abc import ABC, abstractmethod
from datetime import datetime


class DataFeed(ABC):
    _symbols: list[str]

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    @abstractmethod
    def current_datetime(self) -> datetime:
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
