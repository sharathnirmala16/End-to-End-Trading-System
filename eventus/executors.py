import cython
import numpy as np
import pandas as pd

from typing import Type
from abc import ABC, abstractmethod
from common.exceptions import BacktestError
from eventus.strategy import Strategy
from eventus.commissions import Commission
from eventus.datafeeds import HistoricDataFeed
from eventus.brokers import Broker, Backtester


@cython.annotation_typing(True)
@cython.cclass
class Executor(ABC):
    idx: int
    strategy: Strategy
    broker: Broker

    @abstractmethod
    def compute_idx_offset(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def synchronize_indexes(self) -> None:
        pass


@cython.annotation_typing(True)
@cython.cclass
class BacktestExecutor(Executor):
    datafeed: HistoricDataFeed
    equity_curve: list[list]

    def __init__(
        self,
        strategy: Type[Strategy],
        datetime_index: np.ndarray[np.float64],
        data_dict: dict[str, np.ndarray[np.float64]],
        cash: float,
        leverage: float,
        commission_model: Commission,
    ) -> None:
        self.idx = 0
        datafeed = HistoricDataFeed(datetime_index, data_dict)
        self.broker: Backtester = Backtester(cash, leverage, commission_model, datafeed)
        self.strategy = strategy(self.broker, datafeed)

        self.strategy.init()

        self.compute_idx_offset()

    def indicator_offset(self, indicator_name: str) -> int:
        arr = self.strategy.indicators[indicator_name].indicator_data
        res = np.nanargmin(np.isnan(arr))
        if res == 0 and np.isnan(arr[-1]):
            raise BacktestError("Indicator array is empty")
        return res

    def compute_idx_offset(self) -> None:
        self.idx = max(
            self.idx, max(map(self.indicator_offset, self.strategy.indicators.keys()))
        )

    def synchronize_indexes(self) -> None:
        self.broker.idx = self.idx
        self.broker.synchronize_indexes()
        self.strategy.idx = self.idx
        self.strategy.synchronize_indexes()

    def run(self) -> None:
        start, stop = self.idx, self.datafeed.data.shape[0]
        # stop - 1 to ensure an extra row is left for closing any remaining order and positions
        # order of the functions matters as we want orders to be filled on the next tick
        for self.idx in range(start, stop - 1):
            self.synchronize_indexes()
            self.broker.fill_orders()
            self.equity_curve.append(
                [self.datafeed.get_datetime_index()[0], self.broker.current_equity]
            )
            self.strategy.next()

        self.broker.cancel_all_orders()
        self.broker.close_all_positions()

    def results(self) -> dict:
        return self.equity_curve, self.broker.trades
