import pytz
import numpy as np
import pandas as pd

from datetime import datetime
from backtester.datafeed import DataFeed
from backtester.assets_data import AssetsData
from backtester.indicators import Indicator


class BackDataFeed(DataFeed):
    _assets: AssetsData
    _idx: int

    def __init__(self, data_dict: dict[str, pd.DataFrame], symbols: list[str]) -> None:
        super().__init__(symbols)
        self._assets = AssetsData(data_dict)
        self._symbols = self._assets.symbols
        self._indicators = {}
        self._idx = 0

    @property
    def data(self) -> AssetsData:
        return self._assets

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, idx: int) -> None:
        self._idx = idx

    @property
    def indicators(self) -> dict[str, Indicator]:
        return self._indicators

    @property
    def current_datetime(self) -> datetime:
        return datetime.fromtimestamp(self._assets.index[self._idx] / 1e9)

    def bid_price(self, symbol: str) -> float:
        """Set to close price as bid-ask price not given by most vendors as  of now"""
        return self._assets[[symbol, "Close", self._idx]][1]

    def ask_price(self, symbol: str) -> float:
        """Set to close price as bid-ask price not given by most vendors as  of now"""
        return self._assets[[symbol, "Close", self._idx]][1]

    def spot_price(self, symbol: str) -> float:
        """Set to close price as spot price not given by most vendors as  of now"""
        return self._assets[[symbol, "Close", self._idx]][1]

    def add_indicator(self, indicator: Indicator, name: str) -> None:
        self._indicators[name] = indicator

    def __check_key_validity(self, key: int | None) -> None:
        if key is not None and self._idx + key + 1 < 0:
            raise KeyError(
                f"Cannot access values that are further back, current index: {self._idx}"
            )

    def __key_modifier(self, key: int | slice) -> slice:
        if isinstance(key, int):
            self.__check_key_validity(key)
            return self._idx + key + 1

        elif isinstance(key, slice):
            self.__check_key_validity(key.start)
            self.__check_key_validity(key.stop)
            if key.start is not None and key.stop is not None:
                return slice(
                    key.start + self._idx + 1, key.stop + self._idx + 1, key.step
                )

            if key.start is not None and key.stop is None:
                return slice(key.start + self._idx + 1, key.stop, key.step)

            if key.start is None and key.stop is not None:
                return slice(key.start, key.stop + self._idx + 1, key.step)

    def indicator(
        self, symbol: str, indicator_name: str, key: int | slice
    ) -> float | np.ndarray[np.float64]:
        """Meant to be used inside a Strategy's iterator function like next() as the key is linked to index"""
        if indicator_name not in self._indicators:
            raise KeyError(f"{indicator_name} not in {list(self._indicators.keys())}")

        return self._indicators[indicator_name][symbol][self.__key_modifier(key)]

    def price(
        self, symbol: str, price: str, key: int | slice
    ) -> float | np.ndarray[np.float64]:
        """Meant to be used inside a Strategy's iterator function like next() as the key is linked to index"""
        if isinstance(key, int):
            return self._assets[[symbol, price, self.__key_modifier(key)]]

        elif isinstance(key, slice):
            return self._assets[[symbol, price]][self.__key_modifier(key)]
