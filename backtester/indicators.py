import numpy as np
import pandas as pd

from backtester.assets_data import AssetsData
from abc import ABC, abstractmethod


class Indicator(ABC):
    """NOTE: When overloading this class, always
    add the line 'self._compute_indicator(...)'
    at the end of the child class's constructor"""

    _indicator_array: np.ndarray[np.float64]
    _symbols_dict: dict[str, int]
    _assets_data: AssetsData

    def __init__(
        self,
        assets_data: AssetsData,
        symbols: list[str],
        backtesting: bool = True,
        **kwargs,
    ) -> None:
        if backtesting:
            self._assets_data = assets_data
            self._symbols_dict = {
                symbol: (index + 1) for index, symbol in enumerate(symbols)
            }
            self._indicator_array = np.zeros(
                shape=(assets_data.index.shape[0], len(symbols) + 1)
            )
            self.func_kwargs = kwargs
        else:
            raise NotImplementedError("Deployment mode yet to be implemented")

    def _compute_indicator(self) -> None:
        # assigning datetime column to the zeroth column
        self._indicator_array[:, 0] = self._assets_data.index
        for symbol, index in self._symbols_dict.items():
            self._indicator_array[:, index] = self.indicator(symbol, **self.func_kwargs)

    def __getitem__(self, key: int | str | list) -> np.ndarray[np.float64]:
        if isinstance(key, int):
            return self._indicator_array[key]

        elif isinstance(key, str):
            if key in self.symbols:
                sol: np.ndarray[np.float64] = np.zeros(
                    shape=(self._indicator_array.shape[0], 2)
                )
                sol[:, 0] = self._indicator_array[:, 0]
                sol[:, 1] = self._indicator_array[:, self._symbols_dict[key]]
                return sol

            else:
                raise KeyError(f"Invalid key, {key} not found in {self.symbols}")

        elif isinstance(key, list) and len(key) == 2:
            if key[0] in self.symbols and isinstance(key[1], int):
                sol: np.ndarray[np.float64] = np.zeros(shape=(1, 2))
                sol[0, 0] = self._indicator_array[key[1], 0]
                sol[0, 1] = self._indicator_array[key[1], self._symbols_dict[key[0]]]
                return sol[0]

            else:
                raise KeyError(
                    f"Invalid key, {key} should be of type [['symbol', 'int']] with the specified order"
                )

        else:
            raise KeyError(
                """Invalid indexing, these are the possible ways:
                ['int'], ['str'], [['symbol', 'int']]"""
            )

    @property
    def index(self) -> np.ndarray[np.float64]:
        return self._indicator_array[:, 0]

    @property
    def indicator_array(self) -> np.ndarray[np.float64]:
        return self._indicator_array

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols_dict.keys())

    @abstractmethod
    def indicator(self, symbol: str, **kwargs) -> np.ndarray[np.float64]:
        pass


class MovingAverage(Indicator):
    def __init__(
        self,
        assets_data: AssetsData,
        symbols: list[str],
        period: int,
        backtesting: bool = True,
        prices: str = "Close",
        **kwargs,
    ) -> None:
        super().__init__(assets_data, symbols, backtesting, **kwargs)
        self.__period = period
        self.__prices = prices

        # ALWAYS add this line at the end of the child constructor to actually perform the compute
        self._compute_indicator()

    @property
    def period(self) -> str:
        return self.__period

    @property
    def prices(self) -> str:
        return self.__prices

    def indicator(self, symbol: str, **kwargs) -> np.ndarray[np.float64]:
        prices_ser: pd.Series = pd.Series(
            self._assets_data[[symbol, self.__prices]][:, 1]
        )
        return prices_ser.rolling(self.__period).mean().values
