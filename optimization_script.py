import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchanges.nse import Nse
from vendors.yahoo import Yahoo
from credentials import breeze_credentials
from common.enums import INTERVAL

from eventus.datafeeds import HistoricDataFeed
from eventus.indicators import *
from eventus.executors import BacktestExecutor
from eventus.commissions import *
from eventus import strategy
from eventus.analyzer import Analyzer
from eventus.optimizer import optimize
from credentials import breeze_credentials, psql_credentials
from common.enums import *
from securities_master.securities_master import SecuritiesMaster
from securities_master.prices_transformer import PricesTransformer


class MomentumPortfolioStrategy(strategy.Strategy):
    rebalancing_period = 120
    stocks_count = 10
    sl_frac = 0.4

    def momentum_stocks(
        self, data_close: np.ndarray[np.float64], data_vol: np.ndarray[np.float64]
    ) -> set[str]:
        # Filters by momentum and then priority for market cap
        df_close = pd.DataFrame(data_close, columns=self.datafeed.symbols.keys())
        df_vol = pd.DataFrame(data_vol, columns=self.datafeed.symbols.keys())
        returns: dict[str, float] = {}
        mkt_capt: dict[str, float] = {}
        for symbol in df_close.columns:
            mkt_capt[symbol] = df_vol[symbol].iloc[-1] * df_close[symbol].iloc[-1]
            df_close[symbol] = np.log(df_close[symbol].pct_change() + 1)
            returns[symbol] = df_close[symbol].sum()

        final_df = pd.DataFrame(
            {"Returns": returns, "MarketCap": mkt_capt}
        ).sort_values(by=["Returns", "MarketCap"], ascending=False)
        return final_df.index.to_list()[: min(final_df.shape[0], self.stocks_count)]

    def init(
        self,
        rebalancing_period: int = 120,
        stocks_count: int = 10,
        sl_frac: float = 0.4,
    ) -> None:
        self.holding_period: int = 0
        self.rebalancing_period = rebalancing_period
        self.stocks_count = stocks_count
        self.sl_frac = sl_frac

    def next(self) -> None:
        if self.broker.open_positions_count == 0:
            stocks = self.momentum_stocks(
                self.datafeed.get_prices_all_symbols(window=self.rebalancing_period),
                self.datafeed.get_prices_all_symbols(
                    window=self.rebalancing_period, price="Volume"
                ),
            )
            amt = self.broker.margin / len(stocks)
            for stock in stocks:
                size = (
                    amt // self.datafeed.get_prices(stock, price="Close", window=1)[0]
                )
                if size > 0:
                    sl = self.datafeed.get_prices(stock)[0] * (1 - self.sl_frac)
                    self.buy(stock, size=size, sl=sl)
        elif self.holding_period >= self.rebalancing_period:
            for symbol in self.broker.positions:
                self.close_position(symbol)
            self.holding_period = 0
        self.holding_period += 1


def main():
    sm = SecuritiesMaster(
        vendor_login_credentials={
            VENDOR.YAHOO.name: {},
            VENDOR.BREEZE.name: breeze_credentials,
        },
        db_credentials=psql_credentials,
    )

    filter_date = "2006-01-01"
    nse = Nse()
    symbols_df = nse.get_symbols_listing_date("NIFTY200")
    symbols_df = symbols_df[symbols_df["DATE OF LISTING"] < filter_date]

    raw_data_dict = sm.get_prices(
        interval="d1",
        start_datetime=datetime(2006, 1, 1),
        end_datetime=datetime(2018, 1, 1),
        vendor="YAHOO",
        exchange="NSE",
        instrument="STOCK",
        symbols=symbols_df.index.to_list(),
        index=None,
        cache_data=True,
    )

    prices_transformer = PricesTransformer(raw_data_dict, "NSE", "d1")
    prices_transformer.adjust_prices()
    prices_transformer.drop_adj_close()

    start = time.time()
    results = optimize(
        strategy=MomentumPortfolioStrategy,
        datetime_index=prices_transformer.dt_index,
        data_dict=prices_transformer.as_np_data,
        cash=100000,
        leverage=1,
        commission_model=PctFlatCommission(pct=0.05 / 100, amt=20),
        offset=180,
        params={
            "rebalancing_period": range(40, 200, 40),
            "stocks_count": range(5, 25, 5),
            "sl_frac": [0.2, 0.3, 0.4, 0.5],
        },
    ).sort_values(by=["Portfolio Sharpe Ratio", "CAGR (Ann.) [%]"], ascending=False)
    end = time.time()
    results.to_csv("optimization_results.csv", index=True)
    print(f"Execution Time:{end - start}s")


if __name__ == "__main__":
    main()
