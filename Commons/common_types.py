from datetime import datetime


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
