import pandas as pd

from datetime import datetime
from dataclasses import dataclass


@dataclass
class DownloadRequest:
    symbol: str
    start_datetime: datetime
    end_datetime: datetime


class UnprocessedData:
    dataframe: pd.DataFrame | None
    download_requests: list[DownloadRequest] | None

    def __init__(
        self, dataframe: pd.DataFrame | None, download_requests: list[DownloadRequest]
    ) -> None:
        self.data = dataframe
        self.download_requests = download_requests

    def download_required(self) -> bool:
        if self.download_requests is None:
            return False
        return True
