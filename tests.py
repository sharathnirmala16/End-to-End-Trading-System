import pytest
import numpy as np
import pandas as pd
import yfinance as yf

import credentials
from common.enums import *
from common.exceptions import *
from exchanges.nse import Nse
from vendors.vendor import Vendor
from vendors.yahoo import Yahoo
from vendors.breeze import Breeze
from datetime import datetime, timedelta


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
        with pytest.raises(BreezeException):
            self.breeze.get_data(
                interval=INTERVAL.d1,
                exchange=self.nse,
                start_datetime=datetime(2024, 1, 1),
                end_datetime=datetime(2024, 2, 1),
                symbols=symbols,
                adjusted_prices=True,
            )
        # adjusted prices not supported
        with pytest.raises(BreezeException):
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
