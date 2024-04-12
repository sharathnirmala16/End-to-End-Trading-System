import logging.handlers
import os
import queue
import logging
import threading
import importlib
import sqlalchemy
import numpy as np
import pandas as pd
import concurrent.futures

from sqlalchemy import sql, exc
from sqlalchemy.orm import sessionmaker
from vendors.vendor import Vendor
from exchanges.exchange import Exchange
from datetime import datetime, timedelta
from common.exceptions import SecuritiesMasterError
from common.typed_dicts import VENDOR, INTERVAL, INSTRUMENT, EXCHANGE
from securities_master.unprocessed_data import UnprocessedData, DownloadRequest


class SecuritiesMaster:
    __vendor_login_credentials: dict[str, dict[str, str]]

    def __init__(self, vendor_login_credentials: dict[str, dict[str, str]]) -> None:
        # setting up logging
        self.__setup_logger_queue()
        # separate logger thread
        self.__log_thread = threading.Thread(
            target=self.__setup_log_worker, daemon=True
        )
        self.__log_thread.start()
        # log setup done

        self.__vendor_login_credentials = vendor_login_credentials
        self.vendors = VENDOR.create()
        self.exchanges = EXCHANGE.create()
        self.intervals = INTERVAL.create()
        self.instruments = INSTRUMENT.create()

    def __setup_logger_queue(self) -> None:
        # Queue setup
        self.__log_queue = queue.Queue()
        self.__log_queue_handler = logging.handlers.QueueHandler(self.__log_queue)
        self.__logger = logging.getLogger("SecuritiesMasterLog")
        self.__logger.addHandler(self.__log_queue_handler)
        # File Handler Setup
        self.__log_file_handler = logging.FileHandler("logs/securities_master.log")
        self.__log_file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.__logger.setLevel(level=logging.DEBUG)

    def __setup_log_worker(self) -> None:
        listener = logging.handlers.QueueListener(
            self.__log_queue, self.__log_file_handler
        )
        listener.start()

    def __get_ticker_prices_data(
        self,
        ticker: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
    ) -> UnprocessedData:
        table_name: str = (
            f"prices_{ticker.lower()}_{vendor.lower()}_{exchange.lower()}_{interval.lower()}"
        )
        data = pd.DataFrame()
        result = UnprocessedData(None, None)
        try:
            data: pd.DataFrame = pd.read_sql_query(
                sql=f"""
                    SELECT * FROM "{table_name}" 
                    WHERE 
                        "Datetime" >= '{start_datetime.strftime("%Y-%m-%d %H:%M:%S")}' 
                        AND 
                        "Datetime" <= '{end_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
                    ORDER BY
                    "Datetime"
                """,
                con=self.__engine,
            )
            if data.empty:
                raise ValueError
        except (ValueError, exc.ProgrammingError) as e:
            result.download_requests = [
                DownloadRequest(ticker, start_datetime, end_datetime)
            ]

        return result

    def group_download_requests(
        self, unprocessed_data: dict[str, UnprocessedData]
    ) -> list[list[DownloadRequest]] | None:
        grouping_dict: dict[tuple[datetime, datetime], list[DownloadRequest]] = {}

        for unprocessed_request in unprocessed_data.values():
            if unprocessed_request.download_requests is not None:
                for request in unprocessed_request.download_requests:
                    if (
                        request.start_datetime,
                        request.end_datetime,
                    ) not in grouping_dict:
                        grouping_dict.update(
                            {(request.start_datetime, request.end_datetime): [request]}
                        )
                    else:
                        grouping_dict[
                            (request.start_datetime, request.end_datetime)
                        ].append(request)

        if len(grouping_dict) == 0:
            return None

        return [
            [request for request in grouping_dict[start_end_datetimes]]
            for start_end_datetimes in grouping_dict
        ]

    def complete_download_requests(
        self,
        unprocessed_data: dict[str, UnprocessedData],
        exchange_obj: Exchange,
        vendor_obj: Vendor,
        interval: str,
    ) -> tuple[dict[str, pd.DataFrame], set[str]]:
        grouped_requests: list[list[DownloadRequest]] | None = (
            self.group_download_requests(unprocessed_data)
        )

        if grouped_requests is None:
            return {
                ticker: unprocessed_data[ticker].data for ticker in unprocessed_data
            }, set()

        pass

    def get_prices(
        self,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
        instrument: str,
        tickers: list[str] | None = None,
        index: str | None = None,
        cache_data: bool = True,
        progress=False,
    ) -> dict[str, pd.DataFrame]:
        if index == "":
            index = None
        if tickers == []:
            tickers = None

        # checking input validity
        if tickers is None and index is None:
            raise SecuritiesMasterError("Either 'tickers' of 'index' must be defined")
        if vendor not in self.vendors:
            raise SecuritiesMasterError(
                f"'{vendor}' not in vendor list={list(self.vendors.keys())}"
            )
        if exchange not in self.exchanges:
            raise SecuritiesMasterError(
                f"'{exchange}' not in exchange list={list(self.exchanges.keys())}"
            )
        if instrument not in self.instruments:
            raise SecuritiesMasterError(
                f"'{instrument}' not in instrument list={list(self.instruments.keys())}"
            )
        if interval not in self.intervals:
            raise SecuritiesMasterError(
                f"'{interval}' not in interval list={list(self.intervals.keys())}"
            )
        if end_datetime < start_datetime:
            raise SecuritiesMasterError(
                f"start_datetime({start_datetime}) must be before end_datetime({end_datetime})"
            )
        if end_datetime >= datetime.now():
            raise SecuritiesMasterError(
                f"end_datetime({end_datetime}) must be at or before current datetime{datetime.now()}"
            )

        if index is not None and tickers is None:
            exchange_obj: Exchange = getattr(
                importlib.import_module(
                    name=f"exchanges.{exchange.lower()}"
                ),  # module name
                f"{exchange[0]}{exchange[1:].lower()}",  # class name,
            )
            tickers = list(exchange_obj.get_symbols(index).keys())

        vendor_obj: Vendor = getattr(
            importlib.import_module(name=f"vendors.{vendor.lower()}"),  # module name
            f"{vendor[0]}{vendor[1:].lower()}",  # class name
        )(self.__vendor_login_credentials[vendor])

        max_workers = len(tickers) if len(tickers) < os.cpu_count() else os.cpu_count()

        unprocessed_data: dict[str, UnprocessedData] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks = {
                executor.submit(
                    self.__get_ticker_prices_data,
                    ticker,
                    interval,
                    start_datetime,
                    end_datetime,
                    vendor,
                    exchange,
                )
                for ticker in tickers
            }

            for future in concurrent.futures.as_completed(tasks):
                ticker = tasks[future]
                unprocessed_data[ticker] = future.result()

        data_dict: dict[str, pd.DataFrame] = self.complete_download_requests()
