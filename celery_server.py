from typing import Dict, List
from datetime import datetime
from celery_config import celery
from credentials import psql_credentials
from SecuritiesMaster.securities_master import SecuritiesMaster

import time


securities_master = SecuritiesMaster(
    psql_credentials["host"],
    psql_credentials["port"],
    psql_credentials["username"],
    psql_credentials["password"],
)


class Tasks:
    @staticmethod
    @celery.task
    def get_prices_async(
        interval: int,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
        instrument: str,
        vendor_login_credentials: Dict[str, str],
        cache_data: bool,
        index: str | None = None,
        tickers: List[str] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        return securities_master.get_prices(
            interval=interval,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            vendor=vendor,
            exchange=exchange,
            instrument=instrument,
            vendor_login_credentials=vendor_login_credentials,
            cache_data=cache_data,
            index=index,
            tickers=tickers,
        )


if __name__ == "__main__":
    celery.start()
