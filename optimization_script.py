import time
import numpy as np
import pandas as pd
from datetime import datetime
from exchanges.nse import Nse
from credentials import breeze_credentials

from eventus.indicators import *
from eventus.commissions import *
from eventus import strategy
from eventus.optimizer import optimize
from credentials import breeze_credentials, psql_credentials
from common.enums import *
from securities_master.securities_master import SecuritiesMaster
from securities_master.prices_transformer import PricesTransformer


class StdMeanReversion(strategy.Strategy):
    def init(self, devs: float = 4, sl_perc=0.2, tp_perc=0.1):
        self.devs = devs
        self.sl_perc = sl_perc
        self.tp_perc = tp_perc
        self.datafeed.add_indicator("PctChange", PctChange())
        self.symbols_upper_limit = {}
        self.symbols_lower_limit = {}

        for symbol in self.datafeed.symbols:
            arr = self.datafeed.full_symbol_prices(symbol, "PctChange")
            self.symbols_upper_limit[symbol] = np.nanmean(arr) + (
                self.devs * np.nanstd(arr)
            )
            self.symbols_lower_limit[symbol] = np.nanmean(arr) - (
                self.devs * np.nanstd(arr)
            )

    def next(self):
        for symbol in self.datafeed.symbols:
            pct_change = self.datafeed.get_prices(symbol, "PctChange")
            price = self.datafeed.get_prices(symbol, "Close")
            if len(self.broker.positions[symbol]) == 0:
                if pct_change[-1] >= self.symbols_upper_limit[symbol]:
                    self.sell(
                        symbol,
                        sl=price * (1 + self.sl_perc),
                        tp=price * (1 - self.tp_perc),
                    )
                elif pct_change[-1] <= self.symbols_lower_limit[symbol]:
                    self.buy(
                        symbol,
                        sl=price * (1 - self.sl_perc),
                        tp=price * (1 + self.tp_perc),
                    )


def get_symbols() -> list[str]:
    filter_date = "2006-01-01"
    nse = Nse()
    symbols_df = nse.get_symbols_listing_date("NIFTY200")
    symbols_df = symbols_df[symbols_df["DATE OF LISTING"] < filter_date]
    return symbols_df.index.to_list()


def get_transformed_data() -> PricesTransformer:
    # initialize securities_master
    sm = SecuritiesMaster(
        vendor_login_credentials={
            VENDOR.YAHOO.name: {},
            VENDOR.BREEZE.name: breeze_credentials,
        },
        db_credentials=psql_credentials,
    )

    interval: str = "d1"

    raw_data_dict = sm.get_prices(
        interval=interval,
        start_datetime=datetime(2006, 1, 1),
        end_datetime=datetime(2018, 1, 1),
        vendor="YAHOO",
        exchange="NSE",
        instrument="STOCK",
        symbols=get_symbols(),
        index=None,
        cache_data=True,
    )

    prices_transformer = PricesTransformer(raw_data_dict, "NSE", interval)
    prices_transformer.adjust_prices()
    prices_transformer.drop_adj_close()
    return prices_transformer


def main():
    transformed_data: PricesTransformer = get_transformed_data()
    start = time.time()
    results = optimize(
        strategy=StdMeanReversion,
        datetime_index=transformed_data.dt_index,
        data_dict=transformed_data.as_np_data,
        cash=100000,
        leverage=1,
        commission_model=PctFlatCommission(pct=0.05 / 100, amt=20),
        offset=180,
        cols_dict={
            "Open": 0,
            "High": 1,
            "Low": 2,
            "Close": 3,
            "Volume": 4,
            "PctChange": 5,
        },
        params={
            "devs": range(1, 10, 2),
            "sl_perc": np.linspace(0.01, 0.5, num=10),
            "tp_perc": np.linspace(0.01, 0.5, num=10),
        },
    ).sort_values(by=["Portfolio Sharpe Ratio", "CAGR (Ann.) [%]"], ascending=False)
    end = time.time()
    results.to_csv("optimization_results.csv", index=True)
    print(f"Execution Time:{end - start}s")


if __name__ == "__main__":
    main()
