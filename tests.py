import pytest
import numpy as np
import pandas as pd
import yfinance as yf

import credentials
from common.enums import *
from common.exceptions import *
from backtester.commission import *
from exchanges.nse import Nse
from vendors.vendor import Vendor
from vendors.yahoo import Yahoo
from vendors.breeze import Breeze
from datetime import datetime, timedelta
from backtester.assets_data import AssetsData
from backtester.order import Order
from backtester.position import Position
from backtester.trade import Trade
from backtester.commission import *
from backtester.back_datafeed import BackDataFeed
from backtester.indicators import Indicator, MovingAverage


class TestNse:
    def setup_method(self):
        self.nse = Nse()

    def test_get_symbols(self):
        tickers = self.nse.get_symbols(index="NIFTY50")
        assert len(tickers) == 50 and isinstance(tickers, dict)

        with pytest.raises(ValueError):
            df = self.nse.get_symbols(index="EXCEPTION")

    def test_get_symbols_detailed(self):
        data = self.nse.get_symbols_detailed(index="NIFTY50")
        assert data.shape[0] == 50

        with pytest.raises(ValueError):
            df = self.nse.get_symbols_detailed(index="EXCEPTION")


class TestYahoo:
    def setup_method(self):
        self.nse = Nse()
        self.yahoo = Yahoo({})

    @pytest.mark.download_required
    def test_get_adjusted_values(self):
        df_original = yf.download(
            ["RELIANCE.NS"], interval="1d", period="10y", progress=False
        )
        df_adjusted = self.yahoo.get_adjusted_values(df_original)
        original_networth = df_original["Close"] * df_original["Volume"]
        adjusted_networth = df_adjusted["Close"] * df_adjusted["Volume"]

        assert round(original_networth.iloc[0], 2) == round(
            adjusted_networth.iloc[0], 2
        ) and round(original_networth.iloc[-1], 2) == round(
            adjusted_networth.iloc[-1], 2
        )

    @pytest.mark.download_required
    def test_get_data_index_only(self):
        df_orig: pd.DataFrame = yf.download(
            tickers=["^NSEI"],
            start="2024-01-01",
            end="2024-02-01",
            interval="1d",
            progress=False,
        )
        df_my_lib = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 2, 1),
            symbols=["NIFTY50"],
        )

        assert df_orig.equals(df_my_lib["NIFTY50"])

    @pytest.mark.download_required
    def test_get_data_index_constitutents(self):
        data = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 2, 1),
            index="NIFTY50",
        )

        assert len(data) == 50 and data[list(data.keys())[0]].shape[0] > 5

    @pytest.mark.download_required
    def test_get_data_mixed_symbols(self):
        symbols = ["TATAMOTORS", "NIFTY50"]
        data = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 2, 1),
            symbols=symbols,
        )

        assert (
            len(data) == 2
            and list(data.keys()) == symbols
            and data[list(data.keys())[0]].shape[0] > 5
            and data[list(data.keys())[1]].shape[0] > 5
        )

    @pytest.mark.download_required
    def test_get_data_adj_prices_single(self):
        symbols = ["TATAMOTORS"]
        data_adj = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2014, 1, 1),
            end_datetime=datetime(2024, 1, 1),
            symbols=symbols,
            adjusted_prices=True,
        )
        data_orig = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2014, 1, 1),
            end_datetime=datetime(2024, 1, 1),
            symbols=symbols,
            adjusted_prices=False,
        )

        assert (
            data_orig["TATAMOTORS"].shape[1] > data_adj["TATAMOTORS"].shape[1]
            and data_orig["TATAMOTORS"]["Close"].iloc[0]
            != data_adj["TATAMOTORS"]["Close"].iloc[0]
        )

    @pytest.mark.download_required
    def test_get_data_adj_prices_multi(self):
        symbols = ["TATAMOTORS", "RELIANCE"]
        data_adj = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2014, 1, 1),
            end_datetime=datetime(2024, 1, 1),
            symbols=symbols,
            adjusted_prices=True,
        )
        data_orig = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2014, 1, 1),
            end_datetime=datetime(2024, 1, 1),
            symbols=symbols,
            adjusted_prices=False,
        )

        assert (
            data_orig["TATAMOTORS"].shape[1] > data_adj["TATAMOTORS"].shape[1]
            and data_orig["TATAMOTORS"]["Close"].iloc[0]
            != data_adj["TATAMOTORS"]["Close"].iloc[0]
        ) and (
            data_orig["RELIANCE"].shape[1] > data_adj["RELIANCE"].shape[1]
            and data_orig["RELIANCE"]["Close"].iloc[0]
            != data_adj["RELIANCE"]["Close"].iloc[0]
        )

    @pytest.mark.download_required
    def test_get_data_adj_prices_drop_single(self):
        symbols = ["TATAMOTORS"]
        data_adj = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2014, 1, 1),
            end_datetime=datetime(2024, 2, 1),
            symbols=symbols,
            adjusted_prices=True,
        )

        assert "Adj Close" not in data_adj["TATAMOTORS"].columns

    @pytest.mark.download_required
    def test_get_data_adj_prices_drop_multi(self):
        symbols = ["TATAMOTORS", "RELIANCE"]
        data_adj = self.yahoo.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 2, 1),
            symbols=symbols,
            adjusted_prices=True,
        )

        assert (
            "Adj Close" not in data_adj["TATAMOTORS"].columns
            and "Adj Close" not in data_adj["RELIANCE"]
        )

    def test_get_data_errors(self):
        symbols = ["TCS"]
        # no derivable symbols
        with pytest.raises(AttributeError):
            self.yahoo.get_data(
                interval=INTERVAL.y1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                adjusted_prices=True,
            )
        # empty symbols
        with pytest.raises(ValueError):
            self.yahoo.get_data(
                interval=INTERVAL.y1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=[],
                adjusted_prices=True,
            )
        # invalid dates
        with pytest.raises(ValueError):
            self.yahoo.get_data(
                interval=INTERVAL.y1,
                exchange=self.nse,
                symbols=["TCS"],
                start_datetime=datetime(2024, 1, 2),
                end_datetime=datetime(2024, 1, 1),
            )
        # invalid interval
        with pytest.raises(ValueError):
            self.yahoo.get_data(
                interval=INTERVAL.y1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=symbols,
                adjusted_prices=True,
            )

    def test_get_vendor_ticker(self):
        assert self.yahoo.get_vendor_ticker("TCS", self.nse) == "TCS.NS"

    def test_get_symbol_details(self):
        assert len(self.yahoo.get_symbol_details("TCS", self.nse)) > 0


class TestBreeze:
    def setup_method(self):
        # assumes SESSION_TOKEN is defined in credentials.py, if not add it manually to self.login_credentials
        self.login_credentials = credentials.breeze_credentials
        self.nse = Nse()
        self.breeze = Breeze(self.login_credentials)

    def test_customer_details(self):
        assert len(self.breeze.customer_details) > 0

    def test_get_data_errors(self):
        symbols = ["TCS"]
        with pytest.raises(AttributeError):
            # no derivable symbols
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
            )
        # empty symbols list
        with pytest.raises(ValueError):
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=[],
            )
        # invalid dates
        with pytest.raises(ValueError):
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                symbols=symbols,
                start_datetime=datetime(2024, 1, 2),
                end_datetime=datetime(2024, 1, 1),
            )
        # invalid interval
        with pytest.raises(ValueError):
            self.breeze.get_data(
                interval=INTERVAL.y1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=symbols,
            )
        # adjusted prices not supported
        with pytest.raises(BreezeError):
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=symbols,
                adjusted_prices=True,
            )
        # adjusted prices not supported
        with pytest.raises(BreezeError):
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=symbols,
                drop_adjusted_prices=True,
            )

    @pytest.mark.download_required
    def test_get_securities_master(self):
        sec_master = self.breeze.get_securities_master(self.nse)
        assert sec_master.shape[0] > 0

    @pytest.mark.download_required
    def test_get_data_single(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=365 * 2)),
            end_datetime=datetime.today(),
            symbols=["TCS"],
        )["TCS"]
        assert data.shape[0] > 0 and data.shape[1] == 5

    @pytest.mark.download_required
    def test_get_data_single(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=15)),
            end_datetime=datetime.today(),
            symbols=["RELIANCE"],
        )
        assert data["RELIANCE"].shape[0] > 0 and data["RELIANCE"].shape[1] == 5

    @pytest.mark.download_required
    def test_get_data_index_only(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=15)),
            end_datetime=datetime.today(),
            symbols=["NIFTY50"],
        )
        assert data["NIFTY50"].shape[0] > 0 and data["NIFTY50"].shape[1] == 5

    @pytest.mark.download_required
    def test_get_data_index_symbols_mixed(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=15)),
            end_datetime=datetime.today(),
            symbols=["NIFTY50", "TCS"],
        )
        assert data["NIFTY50"].shape[0] > 0 and data["NIFTY50"].shape[1] == 5

    @pytest.mark.download_required
    def test_get_data_index_constituents(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=15)),
            end_datetime=datetime.today(),
            index="NIFTY50",
        )
        assert (
            len(data) == 50
            and data["RELIANCE"].shape[0] > 0
            and data["RELIANCE"].shape[1] == 5
        )

    @pytest.mark.download_required
    def test_get_data_multiple_symbols(self):
        data = self.breeze.get_data(
            interval=INTERVAL.m5,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=15)),
            end_datetime=datetime.today(),
            symbols=["RELIANCE", "ACC", "SBILIFE"],
        )
        assert (
            len(data) == 3
            and data["RELIANCE"].shape[0] > 0
            and data["RELIANCE"].shape[1] == 5
        )

    def test_get_vendor_tickers(self):
        vendor_tickers = self.breeze.get_vendor_tickers(
            symbols=["TCS", "RELIANCE", "TATAMOTORS"], exchange=self.nse
        )
        assert len(vendor_tickers) == 3

    def test_get_vendor_ticker(self):
        assert "TATMOT" == self.breeze.get_vendor_ticker("TATAMOTORS", self.nse)

    def test_get_symbol_details(self):
        assert len(self.breeze.get_symbol_details("TATAMOTORS", exchange=self.nse)) > 0


class TestAssetsDataWithMocks:
    def setup_method(self):
        self.cols = ["Open", "High", "Low", "Close", "Volume"]
        self.tickers = [
            "HDFCBANK.NS",
            "INFY.NS",
            "RELIANCE.NS",
            "TATAMOTORS.NS",
            "TATASTEEL.NS",
            "TCS.NS",
        ]

        self.mock_data = pd.read_csv(
            "mock_data/mock_data.csv", index_col=0, parse_dates=True
        )
        self.mock_data_dict = {ticker: self.mock_data for ticker in self.tickers}
        self.mock_assets_data = AssetsData(self.mock_data_dict)

        self.mock_arr: np.ndarray[np.float64] = np.zeros(shape=(49, 31))
        self.mock_arr[:, 0] = self.mock_data.index.values.astype(np.float64)
        start, end = 1, 6
        for ticker in self.tickers:
            self.mock_arr[:, start:end] = self.mock_data_dict[ticker].values
            start = end
            end += 5

    def dt_arr(self):
        return self.mock_data.index.values.astype(np.float64)

    def test_backtesting_mode_no_data(self):
        with pytest.raises(AttributeError):
            AssetsData()

    def test_constructor(self):
        arr: np.ndarray[np.float64] = self.mock_assets_data.data_array
        assert np.array_equal(arr, self.mock_arr)

    def test_indexing_int(self):
        assert np.array_equal(self.mock_arr[0], self.mock_assets_data[0])
        assert np.array_equal(self.mock_arr[-1], self.mock_assets_data[-1])

    def test_indexing_ticker(self):
        arr: np.ndarray[np.float64] = (
            self.mock_data_dict[self.tickers[0]].reset_index().values
        )
        arr[:, 0] = self.dt_arr()
        assert np.array_equal(arr, self.mock_assets_data[self.tickers[0]])

    def test_indexing_column(self):
        arr: np.ndarray[np.float64] = np.zeros(shape=(49, 7))
        arr[:, 0] = self.dt_arr()
        for i in range(1, len(self.tickers) + 1):
            arr[:, i] = self.mock_data_dict[self.tickers[i - 1]][self.cols[-2]].values
        assert np.array_equal(arr, self.mock_assets_data[self.cols[-2]])

    @pytest.mark.parametrize("col_number", list(range(0, 5)))
    def test_indexing_ticker_column(self, col_number):
        arr: np.ndarray[np.float64] = np.zeros(shape=(49, 2))
        arr[:, 0] = self.dt_arr()
        arr[:, 1] = self.mock_data_dict[self.tickers[0]][self.cols[col_number]].values
        print(self.mock_assets_data["Datetime"])
        assert np.array_equal(
            arr, self.mock_assets_data[[self.tickers[0], self.cols[col_number]]]
        )

    @pytest.mark.parametrize("index", list(range(-1, 5)))
    def test_indexing_ticker_int(self, index):
        arr: np.ndarray[np.float64] = np.zeros(shape=(1, 6))
        arr[:, 0] = self.dt_arr()[index]
        arr[:, 1:] = self.mock_data_dict[self.tickers[0]].values[index]

        assert np.array_equal(arr, self.mock_assets_data[[self.tickers[0], index]])

    @pytest.mark.parametrize("index", list(range(-1, 5)))
    def test_indexing_column_int(self, index):
        arr: np.ndarray[np.float64] = np.zeros(shape=(1, 7))
        arr[:, 0] = self.dt_arr()[index]
        for i in range(1, len(self.tickers) + 1):
            arr[:, i] = self.mock_data_dict[self.tickers[i - 1]][self.cols[0]].values[
                index
            ]

        assert np.array_equal(arr, self.mock_assets_data[[self.cols[0], index]])

    @pytest.mark.parametrize("index", list(range(-1, 5)))
    def test_indexing_ticker_column_int(self, index):
        arr: np.ndarray[np.float64] = np.zeros(shape=(1, 2))
        arr[:, 0] = self.dt_arr()[index]
        arr[:, 1] = self.mock_data_dict[self.tickers[index]][self.cols[index]].values[
            index
        ]
        assert np.array_equal(
            arr[0],
            self.mock_assets_data[[self.tickers[index], self.cols[index], index]],
        )

    def test_index(self):
        assert np.array_equal(self.mock_assets_data.index, self.dt_arr())


class TestOrder:
    def test_size(self):
        with pytest.raises(ValueError):
            Order(
                symbol="TCS",
                order_type=ORDER.BUY,
                size=0,
                placed=datetime.today(),
            )
        with pytest.raises(ValueError):
            Order(
                symbol="TCS",
                order_type=ORDER.BUY,
                size=-1,
                placed=datetime.today(),
            )

    def test_order_type(self):
        with pytest.raises(AttributeError):
            Order(
                symbol="TCS",
                order_type="TEST",
                size=1,
                placed=datetime.today(),
            )

    @pytest.mark.parametrize(
        "sl, tp, order_type", [(1, 0.9, ORDER.BUY), (0.9, 1, ORDER.SELL)]
    )
    def test_market_order_sl_tp(self, sl: float, tp: float, order_type: ORDER):
        with pytest.raises(ValueError):
            Order(
                symbol="TCS",
                order_type=order_type,
                size=1,
                placed=datetime.today(),
                sl=sl,
                tp=tp,
            )

    @pytest.mark.parametrize(
        "sl, price, tp, error",
        [
            (0.1, None, 0.2, AttributeError),
            (None, 0.2, 0.1, ValueError),
            (0.2, 0.1, None, ValueError),
            (0.3, 0.2, 0.1, ValueError),
            (0.2, 0.1, 0.3, ValueError),
            (0.1, 0.3, 0.2, ValueError),
        ],
    )
    def test_limit_order_buy(
        self, sl: float | None, price: float | None, tp: float | None, error: Exception
    ):
        with pytest.raises(error):
            Order(
                symbol="TCS",
                order_type=ORDER.BUY_LIMIT,
                size=1,
                placed=datetime.today(),
                sl=sl,
                price=price,
                tp=tp,
            )

    @pytest.mark.parametrize(
        "sl, price, tp, error",
        [
            (0.1, None, 0.2, AttributeError),
            (None, 0.1, 0.2, ValueError),
            (0.1, 0.2, None, ValueError),
            (0.1, 0.2, 0.3, ValueError),
            (0.3, 0.1, 0.2, ValueError),
            (0.2, 0.3, 0.1, ValueError),
        ],
    )
    def test_limit_order_sell(
        self, sl: float | None, price: float | None, tp: float | None, error: Exception
    ):
        with pytest.raises(error):
            Order(
                symbol="TCS",
                order_type=ORDER.SELL_LIMIT,
                size=1,
                placed=datetime.today(),
                sl=sl,
                price=price,
                tp=tp,
            )


class TestPosition:
    @pytest.mark.parametrize(
        "order, price, placed, commission",
        [
            (None, 10, datetime.today(), 0),
            (
                Order(
                    symbol="TCS", order_type=ORDER.BUY, placed=datetime.today(), price=1
                ),
                None,
                datetime.today(),
                0,
            ),
            (
                Order(
                    symbol="TCS", order_type=ORDER.BUY, placed=datetime.today(), price=1
                ),
                10,
                None,
                0,
            ),
        ],
    )
    def test_exceptions(
        self, order: Order, price: float, placed: datetime, commission: float
    ):
        with pytest.raises(AttributeError):
            Position(order, price, placed, commission)


class TestTrade:

    @pytest.mark.parametrize(
        "open_position, closing_price, closing_datetime, closing_commission",
        [
            (None, 10, datetime.today(), 0),
            (
                Position(
                    Order("TCS", ORDER.BUY, datetime.today(), 1, 10, 9, 11),
                    10,
                    datetime.today(),
                    0,
                ),
                None,
                datetime.today(),
                0,
            ),
            (
                Position(
                    Order("TCS", ORDER.BUY, datetime.today(), 1, 10, 9, 11),
                    10,
                    datetime.today(),
                    0,
                ),
                10,
                None,
                0,
            ),
        ],
    )
    def test_exceptions(
        self,
        open_position: Position,
        closing_price: float,
        closing_datetime: datetime,
        closing_commission: float,
    ):
        with pytest.raises(AttributeError):
            Trade(open_position, closing_price, closing_datetime, closing_commission)

    def test_get_as_dict(self):
        assert {
            "symbol": "TCS",
            "order_type": ORDER.BUY,
            "size": 10,
            "opening_price": 10,
            "closing_price": 10,
            "opening_datetime": datetime.today(),
            "closing_datetime": datetime.today(),
            "commission": 1,
        } == Trade(
            Position(
                Order("TCS", ORDER.BUY, datetime.today(), 10, 10),
                10,
                datetime.today(),
                0.5,
            ),
            10,
            datetime.today(),
            0.5,
        ).get_as_dict()


class TestCommissionModels:
    def test_no_commission(self):
        comm = NoCommission()
        assert 0 == comm.calculate_commission(100, 10)

    def test_flat_commission(self):
        comm = FlatCommission(20)
        assert 20 == comm.calculate_commission(100, 10)

    def test_pct_commission(self):
        with pytest.raises(ValueError):
            PctCommission(-1)
        comm = PctCommission(0.0005)
        assert 100 * 10 * 0.0005 == comm.calculate_commission(100, 10)

    def test_pct_flat_commission(self):
        with pytest.raises(ValueError):
            PctFlatCommission(-1, 20)
        comm = PctFlatCommission(0.0005, 20)
        assert 100 * 10 * 0.0005 == comm.calculate_commission(100, 10)
        assert 20 == comm.calculate_commission(1000, 100)


class TestBackDataFeedWithMocks:
    def setup_method(self):
        self.cols = ["Open", "High", "Low", "Close", "Volume"]
        self.tickers = [
            "HDFCBANK.NS",
            "INFY.NS",
            "RELIANCE.NS",
            "TATAMOTORS.NS",
            "TATASTEEL.NS",
            "TCS.NS",
        ]

        self.mock_data = pd.read_csv(
            "mock_data/mock_data.csv", index_col=0, parse_dates=True
        )
        self.mock_data_dict = {ticker: self.mock_data for ticker in self.tickers}

        self.back_data_feed = BackDataFeed(
            self.mock_data_dict, list(self.mock_data_dict.keys())
        )
        self.mock_assets_data = AssetsData(self.mock_data_dict)

        self.mock_arr: np.ndarray[np.float64] = np.zeros(shape=(49, 31))
        self.mock_arr[:, 0] = self.mock_data.index.values.astype(np.float64)
        start, end = 1, 6
        for ticker in self.tickers:
            self.mock_arr[:, start:end] = self.mock_data_dict[ticker].values
            start = end
            end += 5

        self.ma_indicator = MovingAverage(self.mock_assets_data, self.tickers, period=3)
        self.back_data_feed.add_indicator(self.ma_indicator, name="MA")

    def test_data_property(self):
        assert np.array_equal(
            self.mock_assets_data.data_array, self.back_data_feed.data.data_array
        )

    def test_bid_price(self):
        assert (
            self.back_data_feed.bid_price(self.tickers[-1])
            == self.mock_data_dict[self.tickers[-1]]["Close"].iloc[
                self.back_data_feed.idx
            ]
        )

    def test_ask_price(self):
        assert (
            self.back_data_feed.ask_price(self.tickers[-1])
            == self.mock_data_dict[self.tickers[-1]]["Close"].iloc[
                self.back_data_feed.idx
            ]
        )

    def test_spot_price(self):
        assert (
            self.back_data_feed.spot_price(self.tickers[-1])
            == self.mock_data_dict[self.tickers[-1]]["Close"].iloc[
                self.back_data_feed.idx
            ]
        )

    def test_indicators(self):
        assert self.back_data_feed.indicators == {"MA": self.ma_indicator}

    def test_indicator_error(self):
        pass


class TestIndicator:
    def setup_method(self):
        self.nse = Nse()
        self.vendor = Yahoo({})
        self.symbols = ["HCLTECH", "ACC", "AUROPHARMA"]

        self.data = self.vendor.get_data(
            interval=INTERVAL.d1,
            exchange=self.nse,
            start_datetime=(datetime.today() - timedelta(days=365)),
            end_datetime=datetime.today(),
            symbols=self.symbols,
            adjusted_prices=True,
        )

        self.assets_data = AssetsData(self.data)
        self.indicator = MovingAverage(self.assets_data, self.symbols, period=9)

        for symbol in self.symbols:
            self.data[symbol]["MovingAverage"] = (
                self.data[symbol]["Close"].rolling(9).mean()
            )

    @pytest.mark.parametrize("symbol", ["HCLTECH", "ACC", "AUROPHARMA"])
    def test_indexing_symbol(self, symbol: str):
        assert (
            self.indicator[symbol][:, 1].shape
            == self.data[symbol]["MovingAverage"].values.shape
        )
        assert (
            self.indicator[symbol][:, 1][-1]
            == self.data[symbol]["MovingAverage"].values[-1]
        )

    def test_indexing_int(self):
        arr: np.ndarray[np.float64] = np.zeros(
            shape=(self.assets_data.data_array.shape[0], 4)
        )
        arr[:, 0] = self.assets_data.index
        for i, symbol in enumerate(self.symbols):
            arr[:, i + 1] = self.data[symbol]["MovingAverage"].values

        assert np.array_equal(arr[-1], self.indicator[-1])

    def test_indexing_symbol_int(self):
        arr: np.ndarray[np.float64] = np.zeros(
            shape=(self.assets_data.data_array.shape[0], 4)
        )
        arr[:, 0] = self.assets_data.index
        for i, symbol in enumerate(self.symbols):
            arr[:, i + 1] = self.data[symbol]["MovingAverage"].values

        assert np.array_equal(arr[-1, 0:2], self.indicator[["HCLTECH", -1]])
        assert np.array_equal(
            np.array([arr[-1, 0], arr[-1, 2]]), self.indicator[["ACC", -1]]
        )
        assert np.array_equal(
            np.array([arr[-1, 0], arr[-1, 3]]), self.indicator[["AUROPHARMA", -1]]
        )
