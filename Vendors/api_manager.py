import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Dict, List
from datetime import datetime, timedelta


class APIManager(ABC):
    _login_credentials: Dict[str, str]

    def __init__(self, login_credentials: Dict[str, str]) -> None:
        self._login_credentials = login_credentials

    @staticmethod
    def __search(search_list: list, columns: pd.Index) -> Union[str, None]:
        for element in search_list:
            if element in columns:
                return element
        return None

    @staticmethod
    def process_OHLC_dataframe(
        dataframe: pd.DataFrame,
        datetime_index=True,
        replace_close=False,
        capital_col_names=True,
    ) -> pd.DataFrame:
        df = dataframe.copy(deep=True)
        if datetime_index:
            if df.index.name in ["Date", "date", "Datetime", "datetime"]:
                if capital_col_names:
                    df.index.name = "Datetime"
                else:
                    df.index.name = "datetime"
            elif df.index.name == None:
                col_name = APIManager.__search(
                    ["Date", "date", "Datetime", "datetime"], df.columns
                )
                df.set_index(col_name, inplace=True)
                df.index.name = "datetime"

            if type(df.index[0]) == str:
                df.index = pd.to_datetime(df.index)
        else:
            if df.index.name in ["Date", "date", "Datetime", "datetime"]:
                df = df.reset_index(drop=False)

        cap_names = True if "High" in df.columns else False

        if cap_names and capital_col_names:
            if replace_close and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"].values
        elif cap_names and not capital_col_names:
            if replace_close and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"].values
                df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
        elif not cap_names and capital_col_names:
            if replace_close and "adj close" in df.columns:
                df["close"] = df["adj close"].values
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "adj close": "Adj Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
        else:
            if replace_close and "adj Close" in df.columns:
                df["close"] = df["adj close"].values
        return df

    @staticmethod
    @abstractmethod
    def get_data(
        interval: int,
        exchange: str,
        start_datetime: datetime,
        end_datetime: datetime,
        tickers: List[str] = None,
        index: str = None,
        replace_close=False,
        progress=False,
    ) -> Dict[str, pd.DataFrame]:
        pass

    @staticmethod
    @abstractmethod
    def get_vendor_ticker(ticker: str, exchange: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_ticker_detail(ticker: str, exchange: str, detail: str) -> str:
        pass
