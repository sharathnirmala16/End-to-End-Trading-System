import pandas as pd

from abc import ABC, abstractmethod
from common.enums import INTERVAL
from exchanges.exchange import Exchange
from datetime import datetime


class Vendor(ABC):
    _login_credentials: dict[str, str]
    __supported_intervals: dict[str, str]

    def __init__(self, login_credentials: dict[str, str]) -> None:
        self._login_credentials = login_credentials

    @property
    @abstractmethod
    def supported_intervals(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_data(
        self,
        interval: INTERVAL,
        exchange: Exchange,
        start_datetime: datetime,
        end_datetime: datetime,
        symbols: list[str] = None,
        index: str = None,
        adjusted_prices: bool = False,
        drop_adjusted_prices: bool = False,
    ) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def get_vendor_ticker(self, symbol: str, exchange: Exchange) -> str:
        pass

    @abstractmethod
    def get_symbol_details(self, symbol: str, exchange: Exchange) -> dict:
        pass
