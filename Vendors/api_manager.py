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
