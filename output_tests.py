import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchanges.nse import Nse
from vendors.yahoo import Yahoo
from credentials import breeze_credentials
from common.enums import INTERVAL, ORDER
from backtester.assets_data import AssetsData
from backtester.back_datafeed import BackDataFeed
from backtester.indicators import Indicator, MovingAverage
from backtester.commission import *
from backtester import strategy
from executor import BacktestExecutor

nse = Nse()
vendor = Yahoo(breeze_credentials)

# data = vendor.get_data(
#     interval=INTERVAL.d1,
#     exchange=nse,
#     start_datetime=(datetime.today() - timedelta(days=365)),
#     end_datetime=datetime.today(),
#     symbols=["RELIANCE"],
#     adjusted_prices=True,
# )

data = {
    "RELIANCE": pd.read_csv("nifty50_m1/RELIND.csv", index_col=0, parse_dates=True)[
        :10000
    ],
    "AXISBANK": pd.read_csv("nifty50_m1/AXIBAN.csv", index_col=0, parse_dates=True)[
        :10000
    ],
    "CIPLA": pd.read_csv("nifty50_m1/CIPLA.csv", index_col=0, parse_dates=True)[:10000],
}

assets_data = AssetsData(data)


class MaCrossSignalIndicator(Indicator):
    def __init__(
        self,
        assets_data: AssetsData,
        symbols: list[str],
        sma_period: int,
        lma_period: int,
        backtesting: bool = True,
        prices: str = "Close",
        **kwargs,
    ) -> None:
        super().__init__(assets_data, symbols, backtesting, **kwargs)
        self.__sma_period = sma_period
        self.__lma_period = lma_period
        self.__prices = prices

        # ALWAYS add this line at the end of the child constructor to actually perform the compute
        self._compute_indicator()

    @property
    def sma_period(self) -> str:
        return self.__sma_period

    @property
    def lma_period(self) -> str:
        return self.__lma_period

    @property
    def prices(self) -> str:
        return self.__prices

    def indicator(self, symbol: str, **kwargs) -> np.ndarray[np.float64]:
        prices_ser: pd.Series = pd.Series(
            self._assets_data[[symbol, self.__prices]][:, 1]
        )
        sma = prices_ser.rolling(self.__sma_period).mean()
        lma = prices_ser.rolling(self.__lma_period).mean()
        sma_cross = np.where((sma.shift(1) < lma.shift(1)) & (sma > lma))[0]
        lma_cross = np.where((sma.shift(1) > lma.shift(1)) & (sma < lma))[0]

        signal = np.zeros(len(sma))
        signal[sma_cross] = 1
        signal[lma_cross] = -1

        return signal


# class MaCrossStrategy(strategy.Strategy):
#     sma_period: int = 10
#     lma_period: int = 40
#     sl_perc: float = 3 / 100
#     tp_perc: float = 9 / 100

#     def init(self) -> None:
#         self._data_feed.add_indicator(
#             MaCrossSignalIndicator(
#                 self._data_feed.data,
#                 self._data_feed.symbols,
#                 self.sma_period,
#                 self.lma_period,
#             ),
#             "MACROSS",
#         )


#     def next(self) -> None:
#         symbol = self._data_feed.symbols[0]
#         price = self._data_feed.spot_price(symbol)
#         dt = self._data_feed.current_datetime
#         if len(self._broker.positions) == 0:
#             if (
#                 self._data_feed.indicator(
#                     symbol,
#                     "MACROSS",
#                     -1,
#                 )[1]
#                 == 1
#             ):
#                 size = (self._broker.margin * 0.25) // self._data_feed.spot_price(
#                     symbol
#                 )
#                 self.buy(
#                     symbol=symbol,
#                     order_type=ORDER.BUY,
#                     size=size,
#                     placed=dt,
#                     sl=price * (1 - self.sl_perc),
#                     tp=price * (1 + self.tp_perc),
#                 )
#             elif (
#                 self._data_feed.indicator(
#                     symbol,
#                     "MACROSS",
#                     -1,
#                 )[1]
#                 == -1
#             ):
#                 size = (self._broker.margin * 0.25) // self._data_feed.spot_price(
#                     symbol
#                 )
#                 self.sell(
#                     symbol=symbol,
#                     order_type=ORDER.SELL,
#                     size=size,
#                     placed=dt,
#                     sl=price * (1 + self.sl_perc),
#                     tp=price * (1 - self.tp_perc),
#                 )
class MaCrossStrategy(strategy.Strategy):
    sma_period: int = 10
    lma_period: int = 40
    sl_perc: float = 3 / 100
    tp_perc: float = 9 / 100

    def init(self) -> None:
        self._data_feed.add_indicator(
            MovingAverage(
                self._data_feed.data, self._data_feed.symbols, self.sma_period
            ),
            "SMA",
        )
        self._data_feed.add_indicator(
            MovingAverage(
                self._data_feed.data, self._data_feed.symbols, self.lma_period
            ),
            "LMA",
        )

    def next(self) -> None:
        symbol = self._data_feed.symbols[0]
        price = self._data_feed.spot_price(symbol)
        dt = self._data_feed.current_datetime
        if len(self._broker.positions) == 0:
            if (
                self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -2,
                )[1]
                <= self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -2,
                )[1]
                and self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -1,
                )[1]
                > self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -1,
                )[1]
            ):
                size = (self._broker.margin * 0.25) // self._data_feed.spot_price(
                    symbol
                )
                self.buy(
                    symbol=symbol,
                    order_type=ORDER.BUY,
                    size=size,
                    placed=dt,
                    sl=price * (1 - self.sl_perc),
                    tp=price * (1 + self.tp_perc),
                )
            elif (
                self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -2,
                )[1]
                >= self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -2,
                )[1]
                and self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -1,
                )[1]
                < self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -1,
                )[1]
            ):
                size = (self._broker.margin * 0.25) // self._data_feed.spot_price(
                    symbol
                )
                self.sell(
                    symbol=symbol,
                    order_type=ORDER.SELL,
                    size=size,
                    placed=dt,
                    sl=price * (1 + self.sl_perc),
                    tp=price * (1 - self.tp_perc),
                )


class BackTraderCompStrategy(strategy.Strategy):
    sma_period: int = 10
    lma_period: int = 40
    sl_perc: float = 3 / 100
    tp_perc: float = 9 / 100
    alloc_perc = 25 / 100

    def init(self) -> None:
        self._data_feed.add_indicator(
            MovingAverage(
                self._data_feed.data, self._data_feed.symbols, self.sma_period
            ),
            "SMA",
        )
        self._data_feed.add_indicator(
            MovingAverage(
                self._data_feed.data, self._data_feed.symbols, self.lma_period
            ),
            "LMA",
        )
        self.alloc_amt = self._broker.margin * self.alloc_perc

    def next(self) -> None:
        if (self._broker.positions) == 0:
            self.alloc_amt = self._broker.margin * self.alloc_perc
        for symbol in self._data_feed.symbols:
            price = self._data_feed.spot_price(symbol)
            dt = self._data_feed.current_datetime
            if self.alloc_amt < self._broker.margin:
                if (
                    self._data_feed.indicator(
                        symbol,
                        "SMA",
                        -2,
                    )[1]
                    <= self._data_feed.indicator(
                        symbol,
                        "LMA",
                        -2,
                    )[1]
                    and self._data_feed.indicator(
                        symbol,
                        "SMA",
                        -1,
                    )[1]
                    > self._data_feed.indicator(
                        symbol,
                        "LMA",
                        -1,
                    )[1]
                ):
                    size = self.alloc_amt // price
                    self.buy(
                        symbol=symbol,
                        order_type=ORDER.BUY,
                        size=1,
                        placed=dt,
                    )
            if (
                self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -2,
                )[1]
                >= self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -2,
                )[1]
                and self._data_feed.indicator(
                    symbol,
                    "SMA",
                    -1,
                )[1]
                < self._data_feed.indicator(
                    symbol,
                    "LMA",
                    -1,
                )[1]
            ):
                positions = [
                    position
                    for position in self._broker.positions
                    if position.symbol == symbol
                ]
                if len(positions) > 0:
                    self.close_position(positions[0].position_id)


start = time.time()
bt = BacktestExecutor(
    BackTraderCompStrategy, data, 100000, 1, PctFlatCommission(pct=0.05 / 100, amt=5)
)
res = bt.run(progress_bar=True)
end = time.time()

for _, result in res.items():
    print(result)
print(f"Time taken: {end - start}s")
