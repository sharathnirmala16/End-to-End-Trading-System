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

ma_indicator = MovingAverage(
    assets_data=assets_data,
    symbols=list(data.keys()),
    period=9,
)

feed = BackDataFeed(data, list(data.keys()))
feed.add_indicator(ma_indicator, name="MA")

print(feed.indicator("TCS", "MA", slice(1, 10, 3)).shape[0])
