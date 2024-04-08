import cython
import numpy as np

from abc import ABC, abstractmethod
from eventus.datafeeds import DataFeed


@cython.annotation_typing(True)
@cython.cclass
class Indicator(ABC):
    idx: int
    symbols: dict[str, int]
    datafeed: DataFeed
    indicators_data: np.ndarray[np.float64]

    def __init__(self, datafeed: DataFeed) -> None:
        self.idx = 0
        self.datafeed = datafeed
        self.symbols = datafeed.symbols

        dt_index = self.datafeed.full_datetime_index()
        self.indicator_data = np.zeros(shape=(dt_index.shape[0], len(self.symbols) + 1))

    @abstractmethod
    def indicator(self, symbol: str) -> np.ndarray[np.float64]:
        pass

    def compute_indicator(self) -> None:
        self.indicator_data[:, 0] = self.datafeed.full_datetime_index()
        for symbol in self.symbols:
            self.indicator_data[:, self.symbols[symbol] + 1] = self.indicator(symbol)

    @staticmethod
    def rolling(arr: np.ndarray, window: int) -> np.ndarray:
        shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
        strides = arr.strides + (arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    def full_datetime_index(self) -> np.ndarray[np.float64]:
        return self.indicator_data[:, 0]

    def get_datetime_index(self, window: int = 1) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        return self.indicator_data[max(self.idx - window + 1, 0) : self.idx + 1, 0]

    def full_signal(self, symbol: str) -> np.ndarray[np.float64]:
        return self.indicator_data[
            :,
            self.symbols[symbol] + 1,
        ]

    def get_signal(self, symbol: str, window: int = 1) -> np.ndarray[np.float64]:
        """in-sync with the idx"""
        return self.indicator_data[
            max(self.idx - window + 1, 0) : self.idx + 1,
            self.symbols[symbol] + 1,
        ]


@cython.annotation_typing(True)
@cython.cclass
class MovingAverage(Indicator):
    def __init__(self, datafeed: DataFeed, period: int, price: str = "Close") -> None:
        super().__init__(datafeed)
        self.period: int = period
        self.price: str = price

        self.compute_indicator()

    def indicator(self, symbol: str) -> np.ndarray[np.float64]:
        inp = self.datafeed.full_symbol_prices(symbol, self.price)
        res = np.full_like(inp, np.nan)
        res[self.period - 1 :] = np.mean(
            self.rolling(
                inp,
                window=self.period,
            ),
            axis=1,
        )
        return res


@cython.annotation_typing(True)
@cython.cclass
class ExponentialMovingAverage(Indicator):
    def __init__(self, datafeed: DataFeed, span: int, price: str = "Close") -> None:
        super().__init__(datafeed)
        self.span: int = span
        self.price: str = price

        self.compute_indicator()

    def indicator(self, symbol: str) -> np.ndarray[np.float64]:
        # check citations for the function which was given in stack overflow
        inp = self.datafeed.full_symbol_prices(symbol, self.price)
        res = np.full_like(inp, np.nan)
        alpha = 2 / (self.span + 1)
        alpha_rev = 1 - alpha
        n = inp.shape[0]
        pows = alpha_rev ** (np.arange(n + 1))
        scale_arr = 1 / pows[:-1]
        offset = inp[0] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)
        mult = inp * pw0 * scale_arr
        cumsums = mult.cumsum()
        res[self.span - 1 :] = (offset + cumsums * scale_arr[::-1])[self.span - 1 :]
        return res
