import random
import string
import importlib
import sqlalchemy
import numpy as np
import pandas as pd


from sqlalchemy import sql, exc
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from SecuritiesMaster.sql_commands import commands
from typing import Union, Dict, List
from Exchanges.index_loader import IndexLoader
from Vendors.api_manager import APIManager
from Commons.enums import *


class SecuritiesMaster:
    """
    User can create an instance of this class to obtain data
    """

    def __init__(
        self, host: str, port: int, username: str, password: str, progress=False
    ) -> None:
        """
        Creates the necessary database connection objects.
        """
        try:
            self.__url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/securities_master"
            self.__engine = sqlalchemy.create_engine(
                self.__url, isolation_level="AUTOCOMMIT"
            )
            self.__create_base_tables()
        except Exception as e:
            raise e

    def __create_base_tables(self) -> None:
        """
        Specifically for creating the base tables that are
        necessary for basic operations.
        """
        try:
            for command in commands.values():
                with self.__engine.connect() as conn:
                    conn.execute(sql.text(command))
        except Exception as e:
            raise e

    # NOTE Delete in Final Revision, for testing purpose only
    def temp(self):
        try:
            return self.__get_table_object("symbol")
        except Exception as e:
            raise e

    def get_all_tables(self) -> List[str]:
        try:
            tables = pd.read_sql_query(
                sql="""select table_name from information_schema.tables where table_catalog = 'securities_master' and table_schema = 'public';""",
                con=self.__engine,
            )["table_name"].to_list()
            try:
                tables.remove("users")
                tables.remove("token")
            except:
                pass
            return tables
        except Exception as e:
            raise e

    def get_table(self, table_name: str) -> pd.DataFrame:
        try:
            table = pd.read_sql_table(table_name=table_name, con=self.__engine)
            return table
        except Exception as e:
            raise e

    @staticmethod
    def __get_column_names(table: sqlalchemy.Table) -> List[str]:
        columns_list = []
        for column in table.columns:
            columns_list.append(column.name)

        return columns_list

    def __get_table_object(self, table_name: str) -> sqlalchemy.Table:
        return sqlalchemy.Table(
            table_name, sqlalchemy.MetaData(bind=self.__engine), autoload=True
        )

    def add_row(self, table_name: str, row_data: Dict[str, str]) -> None:
        try:
            table = self.__get_table_object(table_name)
            if "created_datetime" in self.__get_column_names(table):
                row_data["created_datetime"] = datetime.now()
            if "last_updated_datetime" in self.__get_column_names(table):
                row_data["last_updated_datetime"] = datetime.now()
            stmt = sqlalchemy.insert(table).values(row_data)
            with self.__engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            raise e

    def get_row(
        self, table_name: str, primary_key_values: Dict[str, str]
    ) -> Dict[str, str]:
        try:
            table = self.__get_table_object(table_name)
            session = sessionmaker(bind=self.__engine.connect())()
            return dict(session.query(table).filter_by(**primary_key_values).first())
        except Exception as e:
            raise e

    def edit_row(
        self,
        table_name: str,
        old_row_data: Dict[str, str],
        new_row_data: Dict[str, str],
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
            raise e

    def delete_row(self, table_name: str, row_data: Dict[str, str]) -> None:
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
            raise e

    def delete_table(self, table_name: str) -> None:
        try:
            table = self.__get_table_object(table_name)
            table.drop()
        except Exception as e:
            raise e

    def __verify_vendor(self, vendor: str) -> bool:
        vendors = self.get_table("datavendor")["name"].to_list()
        if vendor in vendors:
            return True
        return False

    def __verify_exchange(self, exchange: str) -> bool:
        exchanges = self.get_table("exchange")["name"].to_list()
        if exchange in exchanges:
            return True
        return False

    def __cache_data_to_db(
        self,
        data: pd.DataFrame,
        table_name: str,
        ticker: str,
        vendor: str,
        vendor_obj: APIManager,
        exchange: str,
        interval: int,
        instrument: str,
    ) -> None:
        data.to_sql(name=table_name, con=self.__engine, if_exists="replace", index=True)
        try:
            self.add_row(
                table_name="symbol",
                row_data={
                    "ticker": ticker,
                    "vendor_ticker": vendor_obj.get_vendor_ticker(ticker, exchange),
                    "exchange": exchange,
                    "vendor": vendor,
                    "instrument": INSTRUMENT(instrument).name,
                    "name": ticker,
                    "sector": vendor_obj.get_ticker_detail(ticker, exchange, "sector"),
                    "interval": interval,
                    "linked_table_name": table_name,
                    "created_datetime": (datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_updated_datetime": (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                },
            )
        except:
            self.edit_row(
                table_name="symbol",
                old_row_data=self.get_row(
                    table_name="symbol",
                    primary_key_values={
                        "ticker": ticker,
                        "vendor": vendor,
                        "exchange": exchange,
                        "interval": interval,
                    },
                ),
                new_row_data={
                    "ticker": ticker,
                    "vendor_ticker": vendor_obj.get_vendor_ticker(ticker, exchange),
                    "exchange": exchange,
                    "vendor": vendor,
                    "instrument": INSTRUMENT(instrument).name,
                    "name": ticker,
                    "sector": vendor_obj.get_ticker_detail(ticker, exchange, "sector"),
                    "interval": interval,
                    "linked_table_name": table_name,
                    "created_datetime": (datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_updated_datetime": (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                },
            )

    def __fill_missing_data(
        self,
        dataframe: pd.DataFrame,
        ticker: str,
        interval: int,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        vendor_obj: APIManager,
        exchange: str,
        instrument: str,
        cache_data: bool,
        progress: bool,
    ) -> pd.DataFrame:
        """
        checks dataframe to ensure that all the data in the required date range is present.
        checks to make sure that both the start date and the end date are within 5 days of specified start and end dates.
        if missing, vendor object is used to download the data for the missing date range, if that fails or no data exists,
        dataframe is returned as it is.
        """
        if type(dataframe.index[0]) != pd.Timestamp:
            raise Exception(
                f"Dataframe's index must be of type {type(pd.Timestamp(start_datetime))}, it is of type {type(dataframe.index[0])}"
            )

        table_name: str = f"prices_{ticker.lower()}_{VENDOR(vendor).name.lower()}_{EXCHANGE(exchange).name.lower()}_{INTERVAL(interval).name.lower()}"
        dataframe_start_datetime: datetime = dataframe.index[0].to_pydatetime()
        dataframe_end_datetime: datetime = dataframe.index[-1].to_pydatetime()

        data_appended: bool = False

        if np.abs((dataframe_start_datetime - start_datetime).days) > 5:
            data = vendor_obj.get_data(
                interval=interval,
                exchange=exchange,
                start_datetime=start_datetime - timedelta(days=1),
                end_datetime=dataframe_start_datetime,
                tickers=[ticker],
                replace_close=False,
                progress=False,
            )[ticker]
            dataframe
            index_name = dataframe.index.name
            dataframe = pd.concat([dataframe, data])
            dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
            dataframe.index.name = index_name
            data_appended = True

        if np.abs((dataframe_end_datetime - end_datetime).days) > 5:
            data = vendor_obj.get_data(
                interval=interval,
                exchange=exchange,
                start_datetime=dataframe_end_datetime - timedelta(days=1),
                end_datetime=end_datetime,
                tickers=[ticker],
                replace_close=False,
                progress=False,
            )[ticker]
            index_name = dataframe.index.name
            dataframe = pd.concat([dataframe, data])
            dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
            dataframe.index.name = index_name
            data_appended = True

        dataframe = vendor_obj.process_OHLC_dataframe(
            dataframe=dataframe,
            datetime_index=True,
            replace_close=False,
            capital_col_names=True,
        )

        if data_appended and cache_data:
            self.__cache_data_to_db(
                data=dataframe,
                table_name=table_name,
                ticker=ticker,
                vendor=vendor,
                vendor_obj=vendor_obj,
                exchange=exchange,
                interval=interval,
                instrument=instrument,
            )

        return dataframe

    def __get_ticker_prices_data(
        self,
        ticker: str,
        interval: int,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        vendor_obj: APIManager,
        exchange: str,
        instrument: str,
        cache_data=False,
        progress=False,
    ) -> pd.DataFrame:
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
                con=self.__engine,
            )
            if data.empty:
                raise ValueError
            else:
                data = self.__fill_missing_data(
                    dataframe=vendor_obj.process_OHLC_dataframe(
                        dataframe=data,
                        datetime_index=True,
                        replace_close=False,
                        capital_col_names=True,
                    ).sort_index(ascending=True),
                    ticker=ticker,
                    interval=interval,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    vendor=vendor,
                    vendor_obj=vendor_obj,
                    exchange=exchange,
                    instrument=instrument,
                    cache_data=cache_data,
                    progress=progress,
                )
        except (ValueError, exc.ProgrammingError) as e:
            data = vendor_obj.process_OHLC_dataframe(
                dataframe=vendor_obj.get_data(
                    interval=interval,
                    exchange=exchange,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    tickers=[ticker],
                    replace_close=False,
                    progress=False,
                )[ticker],
                datetime_index=True,
                replace_close=False,
                capital_col_names=True,
            ).sort_index(ascending=True)
            if cache_data:
                self.__cache_data_to_db(
                    data=data,
                    table_name=table_name,
                    ticker=ticker,
                    vendor=vendor,
                    vendor_obj=vendor_obj,
                    exchange=exchange,
                    interval=interval,
                    instrument=instrument,
                )
        return data

    def get_prices(
        self,
        interval: int,
        start_datetime: datetime,
        end_datetime: datetime,
        vendor: str,
        exchange: str,
        instrument: str,
        tickers: List[str] = None,
        index: str = None,
        vendor_login_credentials: Dict[str, str] = {},
        cache_data=False,
        progress=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Publically available method that the user can call to obtain
        data for a list of tickers.
        """
        if index == "":
            index = None
        if tickers == []:
            tickers = None
        # Checking validity of inputs
        if tickers is None and index is None:
            raise Exception("Either 'tickers' of 'index' must be given")
        if not self.__verify_vendor(vendor):
            raise Exception(f"'{vendor}' not in vendor list.")
        if not self.__verify_exchange(exchange):
            raise Exception(f"'{exchange}' not in vendor list.")
        if interval not in [interval.value for interval in INTERVAL]:
            raise Exception(f"{interval} not in INTERVAL Enum.")
        if instrument not in [instrument.value for instrument in INSTRUMENT]:
            raise Exception(f"{instrument} not in INTERVAL Enum.")
        if end_datetime < start_datetime:
            raise Exception(
                f"start_datetime({start_datetime}) must be before end_datetime({end_datetime})"
            )
        if end_datetime >= datetime.now():
            raise Exception(
                f"end_datetime({end_datetime}) must be at or before current datetime{datetime.now()}"
            )

        vendor_obj: APIManager = getattr(
            importlib.import_module(
                name=f"Vendors.{VENDOR(vendor).name.lower()}"
            ),  # module name
            f"{VENDOR(vendor).name[0:1] + VENDOR(vendor).name[1:].lower()}Data",  # class name
        )(vendor_login_credentials)

        if index is not None and tickers is None:
            exchange_obj: IndexLoader = getattr(
                importlib.import_module(
                    name=f"Exchanges.{EXCHANGE(exchange).name.lower()}_tickers"
                ),  # module name
                f"{EXCHANGE(exchange).name}Tickers",  # class name
            )
            tickers = list(exchange_obj.get_tickers(index=index).keys())

        def ticker_prices_wrapper(ticker: str) -> pd.DataFrame:
            return self.__get_ticker_prices_data(
                ticker=ticker,
                interval=interval,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                vendor=vendor,
                vendor_obj=vendor_obj,
                exchange=exchange,
                instrument=instrument,
                cache_data=cache_data,
                progress=progress,
            )

        data_dict: Dict[str, pd.DataFrame] = dict(
            zip(tickers, map(ticker_prices_wrapper, tickers))
        )

        return data_dict
