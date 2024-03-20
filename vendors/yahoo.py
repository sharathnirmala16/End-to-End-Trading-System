import pandas as pd
import yfinance as yf

from enum import Enum
from datetime import datetime
from exchanges.exchange import Exchange
from vendors.vendor import Vendor
from common.enums import INTERVAL


class EXCHANGE_SUFFIX(Enum):
    NSE = "NS"
    BSE = "BO"


class YFINANCE_BENCHMARK_INDEX(Enum):
    NIFTY50 = "^NSEI"
    NIFTY100 = "^CNX100"
    NIFTY500 = "^CRSLDX"
    NIFTYMIDCAP150 = "NIFTYMIDCAP150.NS"
    NIFTYSMALLCAP250 = "NIFTYSMLCAP250.NS"
    NIFTYMICROCAP250 = "NIFTY_MICROCAP250.NS"
    NIFTYNEXT50 = "^NSMIDCP"
    NIFTYBANK = "^NSEBANK"
    NIFTYIT = "^CNXIT"
    NIFTYHEALTHCARE = "NIFTY_HEALTHCARE.NS"
    NIFTYFINSERVICE = "NIFTY_FIN_SERVICE.NS"
    NIFTYAUTO = "^CNXAUTO"
    NIFTYPHARMA = "^CNXPHARMA"
    NIFTYFMCG = "^CNXFMCG"
    NIFTYMEDIA = "^CNXMEDIA"
    NIFTYMETAL = "^CNXMETAL"
    NIFTYREALTY = "^CNXREALTY"


class Yahoo(Vendor):
    def __init__(self, login_credentials: dict[str, str]) -> None:
        super().__init__(login_credentials)
        self.__supported_intervals = {
            INTERVAL.m1.name: "1m",
            INTERVAL.m5.name: "5m",
            INTERVAL.m15.name: "15m",
            INTERVAL.m30.name: "30m",
            INTERVAL.h1.name: "1h",
            INTERVAL.d1.name: "1d",
            INTERVAL.w1.name: "1wk",
            INTERVAL.mo1.name: "1mo",
        }

    @property
    def supported_intervals(self) -> dict[str, str]:
        return self.__supported_intervals

    @staticmethod
    def get_adjusted_values(dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)
        adj_factor = df["Adj Close"] / df["Close"]

        df["Open"] = adj_factor * df["Open"]
        df["High"] = adj_factor * df["High"]
        df["Low"] = adj_factor * df["Low"]
        df["Close"] = adj_factor * df["Close"]
        df["Volume"] = df["Volume"] / adj_factor

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy(deep=True)
        return df

    def get_data(
        self,
        interval: INTERVAL,
        exchange: Exchange,
        start_datetime: datetime,
        end_datetime: datetime,
        symbols: list[str] = None,
        index: str = None,
        adjusted_prices: bool = False,
        drop_adjusted_prices: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Gets the symbols in the index, downloads the data, for each of
        them and processes those that are not empty before returning.\n
        If adjusted_prices=True, then adjusted OHLCV data is returned, else
        OHLCV data and Adj Close kept but dropped if drop_adjusted_prices=True."""
        if symbols is None and index is None:
            raise AttributeError(
                "Either symbols list or an index from which symbols are derived have to be passed"
            )
        if index is not None and symbols is None:
            symbols = exchange.get_symbols(index)

        if len(symbols) < 0:
            raise Exception(
                "Empty symbols list, either failed to obtain symbols from index or passed symbols list is empty"
            )

        if end_datetime < start_datetime:
            raise ValueError(
                f"end_datetime({end_datetime}) cannot be lower that start_datetime({start_datetime})"
            )

        if interval.name not in self.__supported_intervals:
            raise ValueError(
                f"interval({interval.name}) is not supported for Yahoo Finance API."
            )

        interval: str = self.__supported_intervals[interval.name]
        start_date, end_date = start_datetime.strftime(
            "%Y-%m-%d"
        ), end_datetime.strftime("%Y-%m-%d")

        results: dict[str, pd.DataFrame] = {}

        formatted_tickers = {
            symbol: ticker
            for symbol, ticker in zip(
                symbols,
                map(
                    lambda symbol: (
                        f"{symbol}.{EXCHANGE_SUFFIX.__members__[exchange.abbreviation].value}"
                        if symbol not in YFINANCE_BENCHMARK_INDEX._member_names_
                        else YFINANCE_BENCHMARK_INDEX.__members__[symbol].value
                    ),
                    symbols,
                ),
            )
        }

        if len(formatted_tickers) == 1:
            symbol = list(formatted_tickers.keys())[0]
            ticker = formatted_tickers[symbol]
            results[symbol] = yf.download(
                tickers=[ticker],
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            if adjusted_prices:
                results[symbol] = self.get_adjusted_values(results[symbol])
            elif not adjusted_prices and drop_adjusted_prices:
                results[symbol] = results[symbol].drop("Adj Close", axis=1)
        else:
            data: pd.DataFrame = yf.download(
                tickers=list(formatted_tickers.values()),
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for symbol, ticker in formatted_tickers.items():
                results[symbol] = pd.DataFrame()
                for col in cols:
                    results[symbol][col] = data[col][ticker]
                if adjusted_prices:
                    results[symbol] = self.get_adjusted_values(results[symbol])
                elif not adjusted_prices and drop_adjusted_prices:
                    results[symbol] = results[symbol].drop("Adj Close", axis=1)

        return results

    def get_symbol_details(self, symbol: str, exchange: Exchange) -> dict:
        return yf.Ticker(self.get_vendor_ticker(symbol, exchange)).info

    def get_vendor_ticker(self, symbol: str, exchange: Exchange) -> str:
        return f"{symbol}.{EXCHANGE_SUFFIX.__members__[exchange.abbreviation].value}"
