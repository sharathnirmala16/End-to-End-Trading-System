import progressbar
import numpy as np
import pandas as pd

from common.exceptions import *
from backtester.strategy import Strategy
from backtester.broker import Broker
from backtester.back_broker import BackBroker
from backtester.datafeed import DataFeed
from backtester.back_datafeed import BackDataFeed
from backtester.commission import Commission
from backtester.analytics import Analyzer
from typing import Type


class BacktestExecutor:
    __strategy: Type[Strategy]
    __broker: Type[BackBroker]
    __data_feed: BackDataFeed
    __equity_curve: list[list]
    __results: dict
    __idx: int

    def __init__(
        self,
        strategy: Type[Strategy],
        data: dict[str, pd.DataFrame],
        cash: float,
        leverage: float,
        commission: Commission,
    ) -> None:
        self.__idx = 0
        self.__equity_curve = []
        self.__data_feed = BackDataFeed(data_dict=data, symbols=list(data.keys()))
        self.__results = {"Starting Cash [$]": cash}
        self.__broker = BackBroker(
            cash=cash,
            leverage=leverage,
            commission=commission,
            data_feed=self.__data_feed,
        )
        self.__strategy = strategy(self.__broker, self.__data_feed)
        self.__strategy.init()

        self.__compute_idx_offset()

    def __indicator_offset(self, indicator_name: str) -> int:
        arr = self.__data_feed.indicators[indicator_name].indicator_array[:, 1]
        res = np.nanargmin(np.isnan(arr))
        if res == 0 and np.isnan(arr[-1]):
            raise BacktestError("Indicator array is empty")
        return res

    def __compute_idx_offset(self) -> None:
        for indicator_name in self.__data_feed.indicators:
            self.__idx = int(max(self.__idx, self.__indicator_offset(indicator_name)))

    def __synchronize_indexes(self) -> None:
        self.__broker.idx = self.__idx
        self.__data_feed.idx = self.__idx

    def __compute_results(self) -> dict[str, pd.DataFrame]:
        self.__results["Ending Cash [$]"] = self.__broker.cash
        self.__results["Trades"] = self.__broker.trades
        eq_curve = np.array(self.__equity_curve)
        self.__results["Equity Curve [$]"] = pd.Series(
            eq_curve[:, 1], index=eq_curve[:, 0]
        )
        return Analyzer(self.__results).results

    def __run_pgbar(self) -> dict[str, pd.DataFrame]:
        length: int = self.__data_feed.data.data_array.shape[0]
        self.__synchronize_indexes()

        widgets = [
            " [",
            progressbar.Timer(),
            "] ",
            " ",
            progressbar.Percentage(),
            " ",
            progressbar.GranularBar(),
            " ",
            progressbar.AdaptiveETA(),
        ]
        with progressbar.ProgressBar(max_value=length, widgets=widgets) as bar:
            while self.__idx < length - 1:
                self.__equity_curve.append(
                    [self.__data_feed.current_datetime, self.__broker.current_equity]
                )
                self.__broker.close_positions_tp_sl()
                self.__strategy.next()
                self.__idx += 1
                self.__synchronize_indexes()
                self.__broker.fill_orders()
                bar.update(self.idx)
            self.__broker.canel_all_orders()
            self.__broker.close_all_positions()
        return self.__compute_results()

    def __run_no_pgbar(self) -> dict[str, pd.DataFrame]:
        length: int = self.__data_feed.data.data_array.shape[0]
        self.__synchronize_indexes()

        while self.__idx < length - 1:
            self.__equity_curve.append(
                [self.__data_feed.current_datetime, self.__broker.current_equity]
            )
            self.__broker.close_positions_tp_sl()
            self.__strategy.next()
            self.__idx += 1
            self.__synchronize_indexes()
            self.__broker.fill_orders()
        self.__broker.canel_all_orders()
        self.__broker.close_all_positions()

        return self.__compute_results()

    def run(self, progress_bar=False) -> dict[str, pd.DataFrame]:
        if progress_bar:
            return self.__run_pgbar()
        else:
            return self.__run_no_pgbar()

    @property
    def idx(self) -> int:
        return self.__idx
