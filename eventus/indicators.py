import cython
import numpy as np

from copy import deepcopy
from abc import ABC, abstractmethod


@cython.annotation_typing(True)
@cython.cclass
class Indicator(ABC):
    @abstractmethod
    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        pass

    @staticmethod
    def rolling(arr: np.ndarray, window: int) -> np.ndarray:
        inp = deepcopy(arr)
        shape = inp.shape[:-1] + (inp.shape[-1] - window + 1, window)
        strides = inp.strides + (inp.strides[-1],)
        return np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)

    @staticmethod
    def arr_shift(arr: np.ndarray[np.float64], shift: int) -> np.ndarray[np.float64]:
        if shift == 0:
            return arr
        nas = np.empty(abs(shift))
        nas[:] = np.nan
        if shift > 0:
            res = arr[:-shift]
            return np.concatenate((nas, res))
        res = arr[-shift:]
        return np.concatenate((res, nas))


@cython.annotation_typing(True)
@cython.cclass
class MovingAverage(Indicator):
    def __init__(self, period: int) -> None:
        self.period: int = period

    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        inp = deepcopy(arr)
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
    def __init__(self, span: int) -> None:
        self.span: int = span

    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        # check citations for the function which was given in stack overflow
        inp = deepcopy(arr)
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


@cython.annotation_typing(True)
@cython.cclass
class PctChange(Indicator):
    def __init__(self) -> None:
        pass

    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        inp = deepcopy(arr)
        shift_inp = self.arr_shift(inp, 1)
        return (inp / shift_inp) - 1
