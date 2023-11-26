from typing import Dict, List, Tuple
from datetime import datetime
from celery_config import celery
from credentials import psql_credentials
from Commons.enums import *
from Commons.common_types import DownloadRequest
from Commons.common_tasks import CommonTasks
from Vendors.api_manager import APIManager

import pandas as pd
import sqlalchemy


def create_engine():
    try:
        url = f"postgresql+psycopg2://{psql_credentials['username']}:{psql_credentials['password']}@{psql_credentials['host']}:{psql_credentials['port']}/securities_master"
        engine = sqlalchemy.create_engine(url, isolation_level="AUTOCOMMIT")
        return engine
    except Exception as e:
        raise e


engine = create_engine()


class Tasks:
    @staticmethod
    def __get_valid_yahoo_interval(interval: int) -> str:
        interval = INTERVAL(interval).name
        valid_intervals = {
            "m1": "1m",
            "m5": "5m",
            "m15": "15m",
            "m30": "30m",
            "h1": "1h",
            "d1": "1d",
            "w1": "1wk",
            "mo1": "1mo",
            "y1": "1y",
        }

    @staticmethod
    @celery.task
    def get_ticker_prices_data(
        ticker: str,
        interval: int,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
        progress=False,
    ) -> Tuple[pd.DataFrame | None, List[DownloadRequest] | None]:
        table_name: str = f"prices_{ticker.lower()}_{VENDOR(vendor).name.lower()}_{EXCHANGE(exchange).name.lower()}_{INTERVAL(interval).name.lower()}"
        data = pd.DataFrame()
        try:
            data: pd.DataFrame = pd.read_sql_query(
                sql=f"""
                    SELECT * FROM "{table_name}" 
                    WHERE 
                        "Datetime" >= '{start_datetime.strftime("%Y-%m-%d %H:%M:%S")}' 
                        AND 
                        "Datetime" <= '{end_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
                """,
                con=engine,
            )
            if data.empty:
                raise ValueError
            else:
                data = CommonTasks.process_OHLC_dataframe(data)
                return (
                    data,
                    CommonTasks.check_missing_data(
                        ticker, data, start_datetime, end_datetime
                    ),
                )
        except:
            return (None, [DownloadRequest(ticker, start_datetime, end_datetime)])

    @staticmethod
    @celery.task
    def yahoo_single_download(req: DownloadRequest, interval: int) -> pd.DataFrame:
        interval = Tasks.__get_valid_yahoo_interval(interval)
        start_date, end_date = req.start_datetime.strftime(
            "%Y-%m-%d"
        ), req.end_datetime.strftime("%Y-%m-%d")


if __name__ == "__main__":
    celery.start()
