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
from vendors.yahoo import Yahoo
from exchanges.exchange import Exchange
from datetime import datetime, timedelta
from common.exceptions import SecuritiesMasterError
from common.enums import INTERVAL as INTERVAL_ENUM
from common.typed_dicts import VENDOR, INTERVAL, INSTRUMENT, EXCHANGE
from securities_master.sql_commands import commands
from securities_master.unprocessed_data import UnprocessedData, DownloadRequest


class SecuritiesMaster:
    __vendor_login_credentials: dict[str, dict[str, str]]

    def __init__(
        self,
        vendor_login_credentials: dict[str, dict[str, str]],
        db_credentials: dict[str, str],
    ) -> None:
        # setting up logging
        self.__setup_logger_queue()
        # separate logger thread
        self.__log_thread = threading.Thread(
            target=self.__setup_log_worker, daemon=True
        )
        self.__log_thread.start()
        # log setup done

        # initialization of typed_dicts
        self.__vendor_login_credentials = vendor_login_credentials
        self.vendors = VENDOR.create()
        self.exchanges = EXCHANGE.create()
        self.intervals = INTERVAL.create()
        self.instruments = INSTRUMENT.create()

        # yahoo finance for additional data not found with other brokers
        self.__yahoo = Yahoo({})

        # database connection objects
        try:
            self.__url = f"postgresql+psycopg2://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/securities_master"
            self.__engine = sqlalchemy.create_engine(
                self.__url,
                isolation_level="AUTOCOMMIT",
                pool_size=50,
                max_overflow=0,
            )
            self.create_base_tables()
        except Exception as e:
            self.__logger.error(f"__init__(): {e}")
            raise e

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

    def create_base_tables(self) -> None:
        """
        Specifically for creating the base tables that are
        necessary for basic operations.
        """
        try:
            for command in commands.values():
                with self.__engine.connect() as conn:
                    conn.execute(sql.text(command))
        except Exception as e:
            self.__logger.error(f"create_base_tables(): {e}")
            raise e

    def get_all_tables(self) -> list[str]:
        try:
            tables = pd.read_sql_query(
                sql="""select table_name from information_schema.tables where table_catalog = 'securities_master' and table_schema = 'public';""",
                con=self.__engine,
            )["table_name"].to_list()
            try:
                tables.remove("users")
                tables.remove("tokens")
                tables.remove("celery_taskmeta")
                tables.remove("celery_tasksetmeta")
            except:
                pass
            return tables
        except Exception as e:
            self.__logger.error(f"get_all_tables(): {e}")
            raise e

    def get_table(self, table_name: str) -> pd.DataFrame:
        try:
            table = pd.read_sql_table(table_name=table_name, con=self.__engine)
            return table
        except Exception as e:
            self.__logger.error(f"get_table(): {e}")
            raise e

    @staticmethod
    def __get_column_names(table: sqlalchemy.Table) -> list[str]:
        columns_list = []
        for column in table.columns:
            columns_list.append(column.name)

        return columns_list

    def __get_table_object(self, table_name: str) -> sqlalchemy.Table:
        return sqlalchemy.Table(
            table_name, sqlalchemy.MetaData(bind=self.__engine), autoload=True
        )

    def add_row(self, table_name: str, row_data: dict[str, str]) -> None:
        try:
            table = self.__get_table_object(table_name)
            if "created_datetime" in self.__get_column_names(table):
                row_data["created_datetime"] = datetime.now()
            if "last_updated_datetime" in self.__get_column_names(table):
                row_data["last_updated_datetime"] = datetime.now()
            if "time_zone_offset" in self.__get_column_names(table):
                row_data["time_zone_offset"] = "NULL"
            stmt = sqlalchemy.insert(table).values(row_data)
            with self.__engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            self.__logger.error(f"add_row(): {e}")
            raise e

    def get_row(
        self, table_name: str, primary_key_values: dict[str, str]
    ) -> dict[str, str]:
        try:
            table = self.__get_table_object(table_name)
            session = sessionmaker(bind=self.__engine.connect())()
            return dict(session.query(table).filter_by(**primary_key_values).first())
        except Exception as e:
            self.__logger.error(f"get_row(): {e}")
            raise e

    def edit_row(
        self,
        table_name: str,
        old_row_data: dict[str, str],
        new_row_data: dict[str, str],
    ) -> None:
        try:
            table = self.__get_table_object(table_name)
            if "created_datetime" in self.__get_column_names(table):
                new_row_data.pop("created_datetime")
            if "last_updated_datetime" in self.__get_column_names(table):
                new_row_data["last_updated_datetime"] = datetime.now()
            if old_row_data is None:
                raise Exception("old_row_data is None")
            if new_row_data is None:
                raise Exception("new_row_data is None")
            primary_key_values = dict(
                map(lambda col: (col.name, old_row_data[col.name]), table.primary_key)
            )

            stmt = (
                sqlalchemy.update(table)
                .values(new_row_data)
                .where(
                    sqlalchemy.and_(
                        *(
                            getattr(table.c, key) == primary_key_values[key]
                            for key in primary_key_values
                        )
                    )
                )
            )
            with self.__engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            self.__logger.error(f"edit_row(): {e}")
            raise e

    def delete_row(self, table_name: str, row_data: dict[str, str]) -> None:
        try:
            table = self.__get_table_object(table_name)
            primary_key_values = dict(
                map(lambda col: (col.name, row_data[col.name]), table.primary_key)
            )
            stmt = sqlalchemy.delete(table).where(
                sqlalchemy.and_(
                    *(
                        getattr(table.c, key) == primary_key_values[key]
                        for key in primary_key_values
                    )
                )
            )
            with self.__engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            self.__logger.error(f"delete_row(): {e}")
            raise e

    def delete_table(self, table_name: str) -> None:
        try:
            table = self.__get_table_object(table_name)
            table.drop()
        except Exception as e:
            self.__logger.error(f"delete_table(): {e}")
            raise e

    @staticmethod
    def missing_data_download_requests(
        symbol: str,
        dataframe: pd.DataFrame,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[DownloadRequest] | None:
        """
        checks dataframe to ensure that all the data in the required date range is present.
        checks to make sure that both the start date and the end date are within 2 days of specified start and end dates.
        if missing, download request is created and dataframe is not modified.
        """
        if type(dataframe.index[0]) != pd.Timestamp:
            raise Exception(
                f"Dataframe's index must be of type {type(pd.Timestamp(start_datetime))}, it is of type {type(dataframe.index[0])}"
            )

        dataframe_start_datetime: datetime = dataframe.index[0].to_pydatetime()
        dataframe_end_datetime: datetime = dataframe.index[-1].to_pydatetime()

        download_requests: list[DownloadRequest] = []
        if (dataframe_start_datetime - start_datetime).days > 2:
            download_requests.append(
                DownloadRequest(
                    symbol, start_datetime - timedelta(days=1), dataframe_start_datetime
                )
            )

        if (end_datetime - dataframe_end_datetime).days > 2:
            download_requests.append(
                DownloadRequest(
                    symbol, dataframe_end_datetime, end_datetime + timedelta(days=1)
                )
            )

        return download_requests if len(download_requests) != 0 else None

    def __get_ticker_prices_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
    ) -> UnprocessedData:
        table_name: str = (
            f"prices_{symbol.lower()}_{vendor.lower()}_{exchange.lower()}_{interval.lower()}"
        )
        data = pd.DataFrame()
        result = UnprocessedData(None, None)
        try:
            data: pd.DataFrame = pd.read_sql(
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
            else:
                data = data.set_index("Datetime")
                self.__logger.info(
                    f"Read data from Database for {table_name} from {data.index.min()} to {data.index.max()}"
                )
                result.data = data
                result.download_requests = self.missing_data_download_requests(
                    symbol, data, start_datetime, end_datetime
                )
        except (ValueError, exc.ProgrammingError) as e:
            self.__logger.error(f"__get_ticker_prices_data(): {e}")
            result.download_requests = [
                DownloadRequest(symbol, start_datetime, end_datetime)
            ]

        return result

    def group_download_requests(
        self, unprocessed_data: dict[str, UnprocessedData]
    ) -> dict[tuple[datetime, datetime], list[str]]:
        grouped_dict: dict[tuple[datetime, datetime], list[str]] = {}

        for unprocessed_request in unprocessed_data.values():
            if unprocessed_request.download_requests is not None:
                for request in unprocessed_request.download_requests:
                    if (
                        request.start_datetime,
                        request.end_datetime,
                    ) not in grouped_dict:
                        grouped_dict.update(
                            {
                                (request.start_datetime, request.end_datetime): [
                                    request.symbol
                                ]
                            }
                        )
                    else:
                        grouped_dict[
                            (request.start_datetime, request.end_datetime)
                        ].append(request.symbol)

        return grouped_dict

    def complete_download_requests(
        self,
        unprocessed_data: dict[str, UnprocessedData],
        exchange_obj: Exchange,
        vendor_obj: Vendor,
        interval: str,
        adjusted_prices: bool = False,
        drop_adjusted_prices: bool = False,
        **balancing_params,
    ) -> tuple[dict[str, pd.DataFrame], set[str]]:
        grouped_requests: dict[tuple[datetime, datetime], list[str]] = (
            self.group_download_requests(unprocessed_data)
        )

        if len(grouped_requests) == 0:
            return {
                symbol: unprocessed_data[symbol].data for symbol in unprocessed_data
            }, set()

        downloaded_data: dict[str, pd.DataFrame] = {}

        # NOTE: Older code uses common Enums in vendor, which is yet
        # to be updated hence converting interval to equivalent enum
        enum_interval = getattr(INTERVAL_ENUM, interval)
        for requests in grouped_requests:
            downloaded_data.update(
                vendor_obj.get_data(
                    interval=enum_interval,
                    exchange=exchange_obj,
                    start_datetime=requests[0],
                    end_datetime=requests[1],
                    symbols=grouped_requests[requests],
                    index=None,
                    adjusted_prices=adjusted_prices,
                    drop_adjusted_prices=drop_adjusted_prices,
                    **balancing_params,
                )
            )
            self.__logger.info(
                f"Downloaded data for {grouped_requests[requests]} from {requests[0]} to {requests[1]}"
            )

        for symbol in unprocessed_data:
            if symbol in downloaded_data:
                if unprocessed_data[symbol].data is None:
                    unprocessed_data[symbol] = downloaded_data[symbol]
                else:
                    unprocessed_data[symbol] = pd.concat(
                        [unprocessed_data[symbol].data, downloaded_data[symbol]]
                    )
            else:
                unprocessed_data[symbol] = unprocessed_data[symbol].data

        return unprocessed_data, set(downloaded_data.keys())

    def table_exists(self, table_name: pd.DataFrame) -> bool:
        return pd.read_sql(
            f"SELECT EXISTS(SELECT RELNAME FROM PG_CLASS WHERE RELNAME='{table_name}')",
            con=self.__engine,
        ).values[0][0]

    def cache_data_to_db(
        self,
        data: pd.DataFrame,
        symbol: str,
        vendor: str,
        vendor_obj: Vendor,
        exchange: str,
        exchange_obj: Exchange,
        interval: int,
        instrument: str,
    ) -> None:
        table_name: str = (
            f"prices_{symbol.lower()}_{vendor.lower()}_{exchange.lower()}_{interval.lower()}"
        )
        new_data = data.copy(deep=True)
        table_exists = self.table_exists(table_name)

        if table_exists:
            min_date = pd.read_sql(
                f'SELECT MIN("Datetime") FROM {table_name}', con=self.__engine
            ).values[0][0]
            max_date = pd.read_sql(
                f'SELECT MAX("Datetime") FROM {table_name}', con=self.__engine
            ).values[0][0]
            new_data = data[~((data.index <= max_date) & (data.index >= min_date))]

        try:
            new_data.to_sql(
                name=table_name, con=self.__engine, if_exists="append", index=True
            )
            try:
                self.edit_row(
                    table_name="symbol",
                    old_row_data=self.get_row(
                        table_name="symbol",
                        primary_key_values={
                            "ticker": symbol,
                            "vendor": self.vendors[vendor],
                            "exchange": self.exchanges[exchange],
                            "interval": self.intervals[interval],
                        },
                    ),
                    new_row_data={
                        "ticker": symbol,
                        "vendor_ticker": vendor_obj.get_vendor_ticker(
                            symbol, exchange_obj
                        ),
                        "exchange": self.exchanges[exchange],
                        "vendor": self.vendors[vendor],
                        "instrument": self.instruments[instrument],
                        "name": symbol,
                        "sector": self.__yahoo.get_specific_symbol_detail(
                            symbol, exchange_obj, "sector"
                        ),
                        "interval": self.intervals[interval],
                        "linked_table_name": table_name,
                        "created_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                        "last_updated_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                    },
                )
                self.__logger.info(f"{table_name} created in Database")
            except:
                self.add_row(
                    table_name="symbol",
                    row_data={
                        "ticker": symbol,
                        "vendor_ticker": vendor_obj.get_vendor_ticker(
                            symbol, exchange_obj
                        ),
                        "exchange": self.exchanges[exchange],
                        "vendor": self.vendors[vendor],
                        "instrument": self.instruments[instrument],
                        "name": symbol,
                        "sector": self.__yahoo.get_specific_symbol_detail(
                            symbol, exchange_obj, "sector"
                        ),
                        "interval": self.intervals[interval],
                        "linked_table_name": table_name,
                        "created_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                        "last_updated_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                    },
                )
                self.__logger.info(f"{table_name} modified in Database")
            self.__logger.info(f"{table_name} written to Database")
        except Exception as e:
            self.__logger.error(f"cache_data_to_db(): {e}")
            # No need to raise exception as user still gets data, only log error

    def get_prices(
        self,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
        instrument: str,
        symbols: list[str] | None = None,
        index: str | None = None,
        cache_data: bool = False,
        progress=False,
        **balancing_params,
    ) -> dict[str, pd.DataFrame]:
        self.__logger.info("PRICES REQUESTED using get_prices()")
        if index == "":
            index = None
        if symbols == []:
            symbols = None

        # checking input validity
        if symbols is None and index is None:
            raise SecuritiesMasterError("Either 'symbols' of 'index' must be defined")
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

        exchange_obj: Exchange = getattr(
            importlib.import_module(
                name=f"exchanges.{exchange.lower()}"
            ),  # module name
            f"{exchange[0]}{exchange[1:].lower()}",  # class name,
        )()

        if index is not None and symbols is None:
            symbols = list(exchange_obj.get_symbols(index).keys())

        vendor_obj: Vendor = getattr(
            importlib.import_module(name=f"vendors.{vendor.lower()}"),  # module name
            f"{vendor[0]}{vendor[1:].lower()}",  # class name
        )(self.__vendor_login_credentials[vendor])

        max_workers = len(symbols) if len(symbols) < os.cpu_count() else os.cpu_count()

        unprocessed_data: dict[str, UnprocessedData] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks = {
                executor.submit(
                    self.__get_ticker_prices_data,
                    symbol,
                    interval,
                    start_datetime,
                    end_datetime,
                    vendor,
                    exchange,
                ): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(tasks):
                symbol = tasks[future]
                unprocessed_data[symbol] = future.result()

        data_dict, downloaded_symbols = self.complete_download_requests(
            unprocessed_data,
            exchange_obj,
            vendor_obj,
            interval,
            adjusted_prices=False,
            drop_adjusted_prices=False,
            **balancing_params,
        )

        if cache_data and len(downloaded_symbols) != 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                tasks = {
                    executor.submit(
                        self.cache_data_to_db,
                        data_dict[symbol],
                        symbol,
                        vendor,
                        vendor_obj,
                        exchange,
                        exchange_obj,
                        interval,
                        instrument,
                    ): symbol
                    for symbol in downloaded_symbols
                }
                concurrent.futures.wait(tasks)

        self.synchronize_db_symbol_table(vendor_obj, exchange_obj, instrument)

        return data_dict

    def synchronize_db_symbol_table(
        self,
        vendor_obj: Vendor,
        exchange_obj: Exchange,
        instrument: str,
    ) -> None:
        symbol_table = self.get_table("symbol")
        db_price_tables = pd.read_sql(
            sqlalchemy.text(
                "SELECT table_name FROM information_schema.tables  WHERE table_name LIKE 'prices%'"
            ),
            con=self.__engine,
        ).values

        if symbol_table.shape[0] == db_price_tables.shape[0]:
            return

        for table_list in db_price_tables:
            strings = table_list[0].split("_")
            symbol = strings[1].upper()
            vendor = self.vendors[strings[2].upper()]
            exchange = self.exchanges[strings[3].upper()]
            interval = self.intervals[strings[4]]
            try:
                self.get_row(
                    "symbol",
                    {
                        "ticker": symbol,
                        "vendor": vendor,
                        "exchange": exchange,
                        "interval": interval,
                    },
                )
            except:
                self.add_row(
                    table_name="symbol",
                    row_data={
                        "ticker": symbol,
                        "vendor_ticker": vendor_obj.get_vendor_ticker(
                            symbol, exchange_obj
                        ),
                        "exchange": exchange,
                        "vendor": vendor,
                        "instrument": self.instruments[instrument],
                        "name": symbol,
                        "sector": self.__yahoo.get_specific_symbol_detail(
                            symbol, exchange_obj, "sector"
                        ),
                        "interval": interval,
                        "linked_table_name": table_list[0],
                        "created_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                        "last_updated_datetime": (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                    },
                )
