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


from sklearn.linear_model import LogisticRegression


class Indicator1(Indicator):
    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Assuming OHLCV Data is passed in array"""
        return np.where((arr[:, 3] - arr[:, 2]) > (arr[:, 1] - arr[:, 3]), 1, 0)


class Indicator2(Indicator):
    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return np.where(arr > Indicator.arr_shift(arr, 1), 1, 0)


class Indicator3(Indicator):
    def indicator(self, arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return np.where(Indicator.arr_shift(arr, 1) > arr, 1, 0)


class LogRegStrategy(strategy.Strategy):
    def log_reg_model(self, symbol: str) -> LogisticRegression:
        data = pd.DataFrame(
            self.datafeed.get_symbol_all_prices(symbol, window=self.remodelling_period),
            columns=list(self.datafeed.cols_dict.keys()),
        )
        x = data[["C-L > H-C", "H > Ht - 1", "L > Lt - 1"]]
        Y = data["Ct + 1 > C"]

        model = LogisticRegression()
        model.fit(x, Y)
        return model

    @staticmethod
    def make_prediction(
        intercept: float, coeffs: np.ndarray[np.float64], inputs: np.ndarray[np.float64]
    ) -> float:
        z = intercept
        for i in range(inputs.shape[0]):
            z += coeffs[i] + inputs[i]
        z = np.exp(z)
        return z / (1 + z)

    def init(
        self,
        remodelling_period: int = 20,
        prob: float = 0.5,
        order_alloc_frac: float = 0.1,
    ):
        self.remodelling_period = remodelling_period
        self.prob = prob
        self.order_alloc_frac = order_alloc_frac
        self.datafeed.add_indicator_all_prices("C-L > H-C", Indicator1())
        self.datafeed.add_indicator("H > Ht - 1", Indicator2(), price="High")
        self.datafeed.add_indicator("L > Lt - 1", Indicator2(), price="Low")
        self.datafeed.add_indicator("Ct + 1 > C", Indicator3(), price="Close")

        self.models: dict[str, LogisticRegression] = {}
        self.day_count = 0

    def next(self):
        if self.broker.open_positions_count == 0:
            self.max_capital_alloc = self.broker.margin * self.order_alloc_frac

        if self.day_count >= self.remodelling_period:
            self.day_count = 0
            self.models = {}
        elif self.day_count == 0:
            for symbol in self.datafeed.symbols:
                # model creation
                if symbol not in self.models:
                    self.models[symbol] = self.log_reg_model(symbol)

                # initialization of prices, inputs, sizes
                prices = self.datafeed.get_prices(symbol, "Close", 1)
                x = self.datafeed.get_symbol_all_prices(symbol, 1)[5:-1]
                size = self.max_capital_alloc // prices[-1]

                # Entry
                if (
                    len(self.broker.positions[symbol]) == 0
                ) and self.broker.is_fillable_order(symbol, size):
                    y = self.make_prediction(
                        self.models[symbol].intercept_,
                        self.models[symbol].coef_[0],
                        x,
                    )
                    if y > self.prob:
                        self.buy(
                            symbol=symbol,
                            size=size,
                        )
                # Exit
                else:
                    open = self.datafeed.get_prices(symbol, "Open", 1)[-1]
                    prev_close = self.datafeed.get_prices(symbol, "Close", 2)[-2]
                    if open < prev_close:
                        self.close_position(symbol)
            self.day_count += 1
        else:
            self.day_count += 1


def get_symbols() -> list[str]:
    filter_date = "2006-01-01"
    nse = Nse()
    symbols_df = nse.get_symbols_listing_date("NIFTY100")
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
        strategy=LogRegStrategy,
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
            "C-L > H-C": 5,
            "H > Ht - 1": 6,
            "L > Lt - 1": 7,
            "Ct + 1 > C": 8,
        },
        params={
            "remodelling_period": np.linspace(20, 180, num=5).astype(np.int64),
            "prob": np.linspace(0.4, 0.6, num=3),
            "order_alloc_frac": np.linspace(0.1, 0.5, num=4),
        },
    ).sort_values(
        by=[
            "Portfolio Sharpe Ratio",
            "Expected Value [%]",
            "Win Rate [%]",
            "CAGR (Ann.) [%]",
        ],
        ascending=False,
    )
    end = time.time()
    try:
        results.to_csv("optimizations/optimization_log_reg.csv", index=True)
    except:
        results.to_csv(
            f"optimization_results{np.random.randint(1, 100)}.csv", index=True
        )
    print(f"Execution Time:{end - start}s")


if __name__ == "__main__":
    main()
