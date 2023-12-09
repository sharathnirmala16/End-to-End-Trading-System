import pandas as pd

from datetime import datetime
from typing import List


class DownloadRequest:
    ticker: str
    start_datetime: datetime
    end_datetime: datetime

    def __init__(
        self, ticker: str, start_datetime: datetime, end_datetime: datetime
    ) -> None:
        self.ticker = ticker
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime


class UnprocessedData:
    data: pd.DataFrame | None
    download_requests: List[DownloadRequest] | None

    def __init__(
        self, data: pd.DataFrame | None, download_requests: List[DownloadRequest] | None
    ) -> None:
        self.data = data
        self.download_requests = download_requests

    def download_required(self) -> bool:
        if self.download_requests is None:
            return False
        else:
            return True
