import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchanges.nse import Nse
from vendors.yahoo import Yahoo
from credentials import breeze_credentials
from common.enums import INTERVAL
from backtester.assets_data import AssetsData
from backtester.back_datafeed import BackDataFeed
from backtester.indicators import Indicator, MovingAverage
from backtester import strategy

nse = Nse()
vendor = Yahoo(breeze_credentials)

data = vendor.get_data(
    interval=INTERVAL.d1,
    exchange=nse,
    start_datetime=(datetime.today() - timedelta(days=365)),
    end_datetime=datetime.today(),
    symbols=["RELIANCE", "TCS", "TATAMOTORS", "HCLTECH", "ACC", "AUROPHARMA"],
    adjusted_prices=True,
)

assets_data = AssetsData(data)


class MaCrossSignalIndicator(Indicator):
    pass


class MaCrossStrategy(strategy.Strategy):

    def init(self) -> None:
        self.sma = MovingAverage(
            assets_data=assets_data, symbols=assets_data.symbols, period=10
        )
        self.lma = MovingAverage(
            assets_data=assets_data, symbols=assets_data.symbols, period=40
        )
        self._data_feed.add_indicator()

    def next(self) -> None:
        return super().next()
