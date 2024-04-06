import numpy as np

from numba import types
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List
from numba.experimental import jitclass
from common.exceptions import DataFeedError

spec = {
    "idx": types.int64,
    "offset": types.int64,
    "symbols": types.DictType(keyty=types.string, valty=types.int64),
    "data_dict": types.DictType(keyty=types.string, valty=types.float64[:, :]),
    "cols_dict": types.DictType(keyty=types.string, valty=types.int64),
    "data": types.float64[:, :],
}


@jitclass(spec)
class HistoricDataFeed:
    idx: int
    offset: int
    symbols: Dict[str, int]
    data: np.ndarray[np.float64]

    def __init__(
        self,
        datetime_index: np.ndarray[np.float64],
        data_dict: Dict[str, np.ndarray[np.float64]],
    ) -> None:
        self.idx = 0
        self.symbols = Dict.empty(key_type=types.string, value_type=types.int64)

        for index, symbol in enumerate(data_dict.keys()):
            self.symbols[symbol] = index

        self.cols_dict = Dict.empty(key_type=types.string, value_type=types.int64)
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
        # return self.data[
        #     max(self.idx - window + 1, 0) : self.idx + 1,
        #     self.offset * self.symbols[symbol] + self.cols_dict[price],
        # ]
