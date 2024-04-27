import cython
import numpy as np

from numba import types
from abc import ABC, abstractmethod
from eventus.indicators import Indicator
from common.exceptions import DataFeedError

# spec = {
#     "idx": types.int64,
#     "offset": types.int64,
#     "symbols": types.DictType(keyty=types.string, valty=types.int64),
#     "data_dict": types.DictType(keyty=types.string, valty=types.float64[:, :]),
#     "cols_dict": types.DictType(keyty=types.string, valty=types.int64),
#     "data": types.float64[:, :],
# }


# @jitclass(spec)


@cython.annotation_typing(True)
@cython.cclass
class DataFeed(ABC):
    idx: int
    symbols: dict[str, int]

    @abstractmethod
    def full_datetime_index(self) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def full_symbol_prices(
        self, symbol: str, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def full_symbol_all_prices(self, symbol: str) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def full_prices_all_symbols(self, price: str = "Close") -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def get_datetime_index(self, window: int = 1) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def get_prices(
        self, symbol: str, price: str = "Close", window: int = 1
    ) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def get_symbol_all_prices(
        self, symbol: str, window: int = 1
    ) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def get_prices_all_symbols(
        self, window: int = 1, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def add_indicator(
        self, name: str, indicator: Indicator, price: str = "Close"
    ) -> None:
        pass

    @abstractmethod
    def add_indicator_for_symbol(
        self, name: str, indicator: Indicator, symbols: list[str], price: str = "Close"
    ) -> None:
        pass


@cython.annotation_typing(True)
@cython.cclass
class HistoricDataFeed(DataFeed):
    offset: int
    symbols: dict[str, int]
    data: np.ndarray[np.float64]

    def __init__(
        self,
        datetime_index: np.ndarray[np.float64],
        data_dict: dict[str, np.ndarray[np.float64]],
    ) -> None:
        self.idx = 0
        self.symbols = {}

        for index, symbol in enumerate(data_dict.keys()):
            self.symbols[symbol] = index

        self.cols_dict = {}
        self.cols_dict["Datetime"] = 0
        self.cols_dict["Open"] = 1
        self.cols_dict["High"] = 2
        self.cols_dict["Low"] = 3
        self.cols_dict["Close"] = 4
        self.cols_dict["Volume"] = 5

        # shape is rows=number of rows in the first df, cols= 5 (for OHLCV) * number of symbols + 1 (for the Datetime)
        self.offset = len(self.cols_dict) - 1
        self.data = np.zeros(
            shape=(
                data_dict[list(self.symbols.keys())[0]].shape[0],
                self.offset * len(self.symbols) + 1,
            )
        )

        # assiging datetime column to the zeroth column
        self.data[:, 0] = datetime_index

        start, end = 1, self.offset + 1
        for symbol in self.symbols:
            self.data[:, start:end] = data_dict[symbol]
            start = end
            end = start + self.offset

    def full_datetime_index(self) -> np.ndarray[np.float64]:
        return self.data[:, 0]

    def full_symbol_prices(
        self, symbol: str, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        return self.data[
            :,
            self.offset * self.symbols[symbol] + self.cols_dict[price],
        ]

    def full_symbol_all_prices(self, symbol: str) -> np.ndarray[np.float64]:
        return self.data[
            :,
            self.offset * self.symbols[symbol]
            + 1 : self.offset * self.symbols[symbol]
            + self.offset
            + 1,
        ]

    def full_prices_all_symbols(self, price: str = "Close") -> np.ndarray[np.float64]:
        return self.data[:, self.cols_dict[price] :: self.offset]

    def get_datetime_index(self, window: int = 1) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        return self.data[max(self.idx - window + 1, 0) : self.idx + 1, 0]

    def get_prices(
        self, symbol: str, price: str = "Close", window: int = 1
    ) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1,
            self.offset * self.symbols[symbol] + self.cols_dict[price],
        ]

    def get_symbol_all_prices(
        self, symbol: str, window: int = 1
    ) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        col_list: list[str] = list(self.cols_dict.keys())
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1,
            self.offset * self.symbols[symbol]
            + self.cols_dict[col_list[0]]
            + 1 : self.offset * self.symbols[symbol]
            + self.cols_dict[col_list[-1]]
            + 1,
        ]

    def get_prices_all_symbols(
        self, window: int = 1, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1,
            self.cols_dict[price] :: self.offset,
        ]

    # added to allow unit tests to run, deprecated class
    def add_indicator(
        self, name: str, indicator: Indicator, price: str = "Close"
    ) -> None:
        pass

    # added to allow unit tests to run, deprecated class
    def add_indicator_for_symbol(
        self, name: str, indicator: Indicator, symbols: list[str], price: str = "Close"
    ) -> None:
        pass


@cython.annotation_typing(True)
@cython.cclass
class TensorDataFeed(DataFeed):
    symbols: dict[str, int]
    data: np.ndarray[np.float64]
    dt_index: np.ndarray[np.float64]

    def __init__(
        self,
        datetime_index: np.ndarray[np.float64],
        data_dict: dict[str, np.ndarray[np.float64]],
        cols_dict: dict[str, int] = {
            "Open": 0,
            "High": 1,
            "Low": 2,
            "Close": 3,
            "Volume": 4,
        },
    ) -> None:
        self.idx = 0
        self.symbols = {symbol: index for index, symbol in enumerate(data_dict.keys())}
        self.cols_dict = cols_dict
        self.dt_index = datetime_index

        # shape is rows = number of rows in the first, cols = number of cols in cols_dict, height = number of symbols
        self.data = np.zeros(
            shape=(
                data_dict[list(self.symbols.keys())[0]].shape[0],
                len(self.cols_dict),
                len(self.symbols),
            )
        )

        base_columns_size = data_dict[list(self.symbols.keys())[0]].shape[1]
        for index, symbol in enumerate(self.symbols):
            self.data[:, 0:base_columns_size, index] = data_dict[symbol]

    def full_datetime_index(self) -> np.ndarray[np.float64]:
        return self.dt_index

    def full_symbol_prices(
        self, symbol: str, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        return self.data[:, self.cols_dict[price], self.symbols[symbol]]

    def full_symbol_all_prices(self, symbol: str) -> np.ndarray[np.float64]:
        return self.data[:, :, self.symbols[symbol]]

    def full_prices_all_symbols(self, price: str = "Close") -> np.ndarray[np.float64]:
        return self.data[:, self.cols_dict[price], :]

    def get_datetime_index(self, window: int = 1) -> np.ndarray[np.float64]:
        return self.dt_index[max(self.idx - window + 1, 0) : self.idx + 1]

    def get_prices(
        self, symbol: str, price: str = "Close", window: int = 1
    ) -> np.ndarray[np.float64]:
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1,
            self.cols_dict[price],
            self.symbols[symbol],
        ]

    def get_symbol_all_prices(
        self, symbol: str, window: int = 1
    ) -> np.ndarray[np.float64]:
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1, :, self.symbols[symbol]
        ]

    def get_prices_all_symbols(
        self, window: int = 1, price: str = "Close"
    ) -> np.ndarray[np.float64]:
        return self.data[
            max(self.idx - window + 1, 0) : self.idx + 1, self.cols_dict[price], :
        ]

    def add_indicator(
        self, name: str, indicator: Indicator, price: str = "Close"
    ) -> None:
        # for some reason fails horibly with np.apply_along_axis
        for symbol in self.symbols:
            self.data[:, self.cols_dict[name], self.symbols[symbol]] = (
                indicator.indicator(
                    self.data[:, self.cols_dict[price], self.symbols[symbol]]
                )
            )

    def add_indicator_for_symbol(
        self, name: str, indicator: Indicator, symbols: list[str], price: str = "Close"
    ) -> None:
        for symbol in symbols:
            self.data[:, self.cols_dict[name], self.symbols[symbol]] = (
                indicator.indicator(
                    arr=self.data[:, self.cols_dict[price], self.symbols[symbol]]
                )
            )
