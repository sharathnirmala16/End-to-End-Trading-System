import pandas as pd

from backtester.strategy import Strategy
from backtester.broker import Broker
from backtester.datafeed import DataFeed
from backtester.back_datafeed import BackDataFeed
from backtester.assets_data import AssetsData
from backtester.commission import Commission
from typing import Type


# NOTE:Consider backtester executor and deployment executor as separate executors?
class Executor:
    __strategy: Type[Strategy]
    __broker: Type[Broker]
    __data_feed: DataFeed
    __idx: int
    __backtesting: bool

    def __init__(
        self,
        strategy: Type[Strategy],
        broker: Type[Broker],
        data: dict[str, pd.DataFrame],
        cash: float,
        leverage: float,
        commission: Commission,
        backtesting: bool,
    ) -> None:
        self.__idx = 0
        self.__backtesting = backtesting
        if backtesting:
            self.__data_feed = BackDataFeed(data_dict=data, symbols=list(data.keys()))
            self.__broker = broker(
                cash=cash,
                leverage=leverage,
                commission=commission,
                data_feed=self.__data_feed,
            )
            self.__strategy = strategy(self.__broker)
            self.__strategy.init()
            self.__compute_idx_offset()
        else:
            raise NotImplementedError("Deployment yet to be implemented")

    def __compute_idx_offset(self) -> None:
        pass

    @property
    def idx(self) -> int:
        return self.__idx
