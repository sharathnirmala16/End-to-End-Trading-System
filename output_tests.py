import time
import cython
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchanges.nse import Nse
from vendors.yahoo import Yahoo
from credentials import breeze_credentials
from common.enums import INTERVAL

from numba import types
from numba.typed.typeddict import Dict
from eventus.datafeeds import HistoricDataFeed, TensorDataFeed
from eventus.indicators import *
from eventus.executors import BacktestExecutor
from eventus.commissions import *
from eventus import strategy

nse = Nse()
vendor = Yahoo(breeze_credentials)
symbols = ["RELIANCE", "TCS", "TATAMOTORS", "HCLTECH", "ACC", "AUROPHARMA"]
# data = vendor.get_data(
#     interval=INTERVAL.d1,
#     exchange=nse,
#     start_datetime=(datetime.today() - timedelta(days=365)),
#     end_datetime=datetime.today(),
#     symbols=["RELIANCE", "TCS", "TATAMOTORS", "HCLTECH", "ACC", "AUROPHARMA"],
#     adjusted_prices=True,
#     drop_adjusted_prices=True,
# )
data = {
    "RELIANCE": pd.read_csv("nifty50_m1/RELIND.csv", index_col=0, parse_dates=True),
    "AXISBANK": pd.read_csv("nifty50_m1/AXIBAN.csv", index_col=0, parse_dates=True),
    "CIPLA": pd.read_csv("nifty50_m1/CIPLA.csv", index_col=0, parse_dates=True),
}
# data = {
#     "RELIANCE": pd.read_csv("mock_data/mock_data2.csv", index_col=0, parse_dates=True),
#     "AXISBANK": pd.read_csv("mock_data/mock_data2.csv", index_col=0, parse_dates=True),
#     "CIPLA": pd.read_csv("mock_data/mock_data2.csv", index_col=0, parse_dates=True),
# }


dt_index = data["RELIANCE"].index.values.astype(np.float64)
np_data = Dict.empty(key_type=types.string, value_type=types.float64[:, :])
for symbol in data:
    np_data[symbol] = data[symbol].values.astype(np.float64)

datafeed = TensorDataFeed(dt_index, np_data)

print(data.keys())
print(data["CIPLA"].shape)


@cython.annotation_typing(True)
@cython.cclass
class BackTraderCompStrategy(strategy.Strategy):
    sma_period: int = 10
    lma_period: int = 40

    def init(self) -> None:
        self.datafeed.add_indicator("SMA", MovingAverage(self.sma_period))
        self.datafeed.add_indicator("LMA", MovingAverage(self.lma_period))

        # self.alloc_amt = self.broker.margin * self.alloc_perc

    def next(self) -> None:
        # if self.broker.open_positions_count == 0:
        #     self.alloc_amt = self.broker.margin * self.alloc_perc

        for symbol in self.datafeed.symbols:
            sma = self.datafeed.get_prices(symbol, "SMA", 2)
            lma = self.datafeed.get_prices(symbol, "LMA", 2)
            if len(self.broker.positions[symbol]) == 0:
                if sma[0] <= lma[0] and sma[1] > lma[1]:
                    self.buy(symbol)

            if sma[0] >= lma[0] and sma[1] < lma[1]:
                self.close_position(symbol)


start = time.time()
bt = BacktestExecutor(
    strategy=BackTraderCompStrategy,
    datetime_index=dt_index,
    data_dict=np_data,
    cash=100000.0,
    leverage=1.0,
    commission_model=PctFlatCommission(pct=0.05 / 100, amt=5),
    cols_dict={
        "Open": 0,
        "High": 1,
        "Low": 2,
        "Close": 3,
        "Volume": 4,
        "SMA": 5,
        "LMA": 6,
    },
)
bt.run(progress=True)
end = time.time()
res = bt.results()
print(res["KPIs"])
print(f"Execution Time: {end - start}s")
