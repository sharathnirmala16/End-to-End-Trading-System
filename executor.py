import numpy as np
import pandas as pd

from common.exceptions import *
from backtester.strategy import Strategy
from backtester.broker import Broker
from backtester.back_broker import BackBroker
from backtester.datafeed import DataFeed
from backtester.back_datafeed import BackDataFeed
from backtester.commission import Commission
from typing import Type


# NOTE:Consider backtester executor and deployment executor as separate executors?
class BacktestExecutor:
    __strategy: Type[Strategy]
    __broker: Type[BackBroker]
    __data_feed: BackDataFeed
    __results: dict
    __idx: int

    def __init__(
        self,
        strategy: Type[Strategy],
        broker: Type[BackBroker],
        data: dict[str, pd.DataFrame],
        cash: float,
        leverage: float,
        commission: Commission,
    ) -> None:
        self.__idx = 0
        self.__data_feed = BackDataFeed(data_dict=data, symbols=list(data.keys()))
        self.__results["Starting Cash [$]"] = cash
        self.__broker = broker(
            cash=cash,
            leverage=leverage,
            commission=commission,
            data_feed=self.__data_feed,
        )
        self.__strategy = strategy(self.__broker)
        self.__strategy.init()

        self.__compute_idx_offset()

    def __compute_idx_offset(self) -> None:
        arr = self.__data_feed.indicators[self.__data_feed.symbols].indicator_array
        while arr[self.__idx, 1] == np.nan and self.__idx <= arr.shape[0] - 1:
            self.__idx += 1
        if self.__idx == arr.shape[0] - 1:
            raise BacktestError("Indicator array is empty")

    def __synchronize_indexes(self) -> None:
        self.__broker.idx = self.__idx
        self.__data_feed.idx = self.__idx

    def __compute_results(self) -> dict:
        self.__results["Ending Cash [$]"] = self.__broker.cash
        self.__results["Trades"] = self.__broker.trades
        return self.__results

    def run(self) -> dict:
        length: int = self.__data_feed.indicators[
            self.__data_feed.symbols
        ].indicator_array.shape[0]
        while self.__idx < length - 1:
            self.__broker.close_positions_tp_sl()
            self.__strategy.next()
            self.__idx += 1
            self.__synchronize_indexes()
            self.__broker.fill_orders()
        self.__broker.canel_all_orders()
        self.__broker.close_all_positions()
        return self.__compute_results()

    @property
    def idx(self) -> int:
        return self.__idx
