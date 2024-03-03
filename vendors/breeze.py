import os
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import concurrent.futures

from datetime import datetime
from breeze_connect import BreezeConnect
from exchanges.exchange import Exchange
from vendors.vendor import Vendor
from common.enums import INTERVAL
from common.exceptions import BreezeException


class Breeze(Vendor):
    def __init__(self, login_credentials: dict[str, str]) -> None:
        super().__init__(login_credentials)
        self.__supported_intervals = {
            INTERVAL.m1.name: "1minute",
            INTERVAL.m5.name: "5minute",
            INTERVAL.m30.name: "30minute",
            INTERVAL.d1.name: "1day",
        }
        self.__supported_indexes = {
            "NIFTY50": "NIFTY",
            "NIFTYBANK": "CNXBAN",
            "NIFTYFINSERVICE": "NIFFIN",
            "NIFTYIT": "CNXIT",
        }
        self.__breeze = BreezeConnect(api_key=self._login_credentials["API_KEY"])
        self.__breeze.generate_session(
            api_secret=self._login_credentials["SECRET_KEY"],
            session_token=self._login_credentials["SESSION_TOKEN"],
        )

    @property
    def supported_intervals(self) -> dict[str, str]:
        return self.__supported_intervals

    @property
    def customer_details(self) -> dict:
        details: dict = self.__breeze.get_customer_details(
            api_session=self._login_credentials["SESSION_TOKEN"]
        )
        if details["Status"] >= 200 and details["Status"] <= 299:
            return details
        else:
            raise BreezeException(details["Error"])

    @property
    def breeze(self):
        return self.__breeze

    @property
    def supported_indexes(self) -> dict[str, str]:
        return self.__supported_indexes

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "close": "Close",
                "datetime": "Datetime",
                "high": "High",
                "low": "Low",
                "open": "Open",
                "volume": "Volume",
            }
        )
        df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy(deep=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
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
        balance_dataframes: bool = True,
    ) -> dict[str, pd.DataFrame]:
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

        if adjusted_prices or drop_adjusted_prices:
            raise BreezeException("Adjusted Prices are not supported by Breeze yet.")

        start_date, end_date = start_datetime.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ), end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        results: dict[str, pd.DataFrame] = {}

        formatted_tickers = self.get_vendor_tickers(symbols, exchange)

        if len(formatted_tickers) == 1:
            symbol = list(formatted_tickers.keys())[0]
            ticker = formatted_tickers[symbol]
            results[symbol] = self.preprocess_dataframe(
                pd.DataFrame(
                    self.__breeze.get_historical_data(
                        interval=self.__supported_intervals[interval.name],
                        from_date=start_date,
                        to_date=end_date,
                        stock_code=ticker,
                        exchange_code=exchange.abbreviation,
                        product_type="cash",
                    )["Success"]
                )
            )
        else:
            max_workers = min(os.cpu_count(), len(symbols))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                results = {
                    symbol: executor.submit(
                        (
                            lambda ticker: self.__breeze.get_historical_data(
                                interval=self.__supported_intervals[interval.name],
                                from_date=start_date,
                                to_date=end_date,
                                stock_code=ticker,
                                exchange_code=exchange.abbreviation,
                                product_type="cash",
                            )
                        ),
                        ticker,
                    )
                    for symbol, ticker in formatted_tickers.items()
                }

            for symbol, future in results.items():
                # done to ensure no miss
                res = future.result()
                time.sleep(0.01)
                results[symbol] = self.preprocess_dataframe(
                    pd.DataFrame(res["Success"])
                )

            if balance_dataframes:
                results = self.balance_dataframes(results, interval)

        return results

    @staticmethod
    def balance_dataframes(
        results: dict[str, pd.DataFrame],
        interval: INTERVAL,
        interpolation_limit: int = 2,
        filter_time: bool = True,
        filter_start_time="9:15",
        filter_end_time="15:30",
    ) -> dict[str, pd.DataFrame]:
        combined_index = results[list(results.keys())[0]].index
        for symbol in results:
            results[symbol] = results[symbol][~results[symbol].index.duplicated()]
            combined_index = pd.Index.union(combined_index, results[symbol].index)
        combined_index = combined_index.drop_duplicates()

        for symbol in results:
            results[symbol] = (
                results[symbol]
                .apply(pd.to_numeric, errors="coerce")
                .reindex(combined_index, fill_value=np.nan)
                .interpolate(limit_direction="both", limit=interpolation_limit)
            )

        if filter_time and interval != INTERVAL.d1:
            for symbol in results:
                results[symbol] = results[symbol].between_time(
                    pd.Timestamp(filter_start_time).time(),
                    pd.Timestamp(filter_end_time).time(),
                )
        return results

    def get_vendor_tickers(
        self, symbols: list[str], exchange: Exchange
    ) -> dict[str, str]:
        sec_master = (
            Breeze.get_securities_master(exchange)
            .dropna(subset=["ISINCode"])
            .set_index("ISINCode")
        )
        detailed_symbols = exchange.get_symbols_detailed(index="ALL_TICKERS").set_index(
            "SYMBOL"
        )
        vendor_tickers: dict[str, str] = {
            symbol: ticker
            for symbol, ticker in zip(
                symbols,
                map(
                    lambda symbol: (
                        sec_master.loc[detailed_symbols.loc[symbol]["ISIN NUMBER"]][
                            "ShortName"
                        ]
                        if symbol not in self.__supported_indexes
                        else self.__supported_indexes[symbol]
                    ),
                    symbols,
                ),
            )
        }

        return vendor_tickers

    @staticmethod
    def get_securities_master(exchange: Exchange, brief=True) -> pd.DataFrame:
        url = "https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip"
        if exchange.abbreviation == "NSE":
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open("temp.zip", "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            with zipfile.ZipFile("temp.zip", "r") as zip_ref:
                csv_file = zip_ref.extract(
                    zip_ref.namelist()[zip_ref.namelist().index("NSEScripMaster.txt")]
                )
            df = pd.read_csv(csv_file, sep=",")
            df.columns = [col.strip('" ').strip('"') for col in df.columns]
            os.remove("temp.zip")
            os.remove(csv_file)
            if brief:
                return df[["ISINCode", "ShortName", "CompanyName"]]
            else:
                return df

        elif exchange.abbreviation == "BSE":
            raise NotImplementedError("Yet to implement BSE")

    def get_symbol_details(self, symbol: str, exchange: Exchange) -> dict:
        sec_master = (
            Breeze.get_securities_master(exchange, brief=False)
            .dropna(subset=["ISINCode"])
            .set_index("ISINCode")
        )
        detailed_symbols = exchange.get_symbols_detailed(index="ALL_TICKERS").set_index(
            "SYMBOL"
        )
        return sec_master.loc[detailed_symbols.loc[symbol]["ISIN NUMBER"]].to_dict()

    def get_vendor_ticker(self, symbol: str, exchange: Exchange) -> str:
        sec_master = (
            Breeze.get_securities_master(exchange)
            .dropna(subset=["ISINCode"])
            .set_index("ISINCode")
        )
        detailed_symbols = exchange.get_symbols_detailed(index="ALL_TICKERS").set_index(
            "SYMBOL"
        )
        return sec_master.loc[detailed_symbols.loc[symbol]["ISIN NUMBER"]]["ShortName"]
