from typing import Dict, List, Tuple
from datetime import datetime
from celery_config import celery
from credentials import psql_credentials
from Commons.enums import *
from Commons.common_types import DownloadRequest
from Commons.common_tasks import CommonTasks
from Vendors.api_manager import APIManager
from sqlalchemy.orm import sessionmaker

import pandas as pd
import sqlalchemy
import importlib


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
    def __get_column_names(table: sqlalchemy.Table) -> List[str]:
        columns_list = []
        for column in table.columns:
            columns_list.append(column.name)

        return columns_list

    def __get_table_object(table_name: str) -> sqlalchemy.Table:
        return sqlalchemy.Table(
            table_name, sqlalchemy.MetaData(bind=engine), autoload=True
        )

    @staticmethod
    def add_row(table_name: str, row_data: Dict[str, str]) -> None:
        try:
            table = Tasks.__get_table_object(table_name)
            if "created_datetime" in Tasks.__get_column_names(table):
                row_data["created_datetime"] = datetime.now()
            if "last_updated_datetime" in Tasks.__get_column_names(table):
                row_data["last_updated_datetime"] = datetime.now()
            if "time_zone_offset" in Tasks.__get_column_names(table):
                row_data["time_zone_offset"] = "NULL"
            stmt = sqlalchemy.insert(table).values(row_data)
            with engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            raise e

    @staticmethod
    def edit_row(
        table_name: str,
        old_row_data: Dict[str, str],
        new_row_data: Dict[str, str],
    ) -> None:
        try:
            table = Tasks.__get_table_object(table_name)
            if "created_datetime" in Tasks.__get_column_names(table):
                new_row_data.pop("created_datetime")
            if "last_updated_datetime" in Tasks.__get_column_names(table):
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
            with engine.connect() as conn:
                conn.execute(stmt)
        except Exception as e:
            raise e

    @staticmethod
    def get_row(table_name: str, primary_key_values: Dict[str, str]) -> Dict[str, str]:
        try:
            table = Tasks.__get_table_object(table_name)
            session = sessionmaker(bind=engine.connect())()
            return dict(session.query(table).filter_by(**primary_key_values).first())
        except Exception as e:
            raise e

    # @staticmethod
    # def __get_valid_yahoo_interval(interval: int) -> str:
    #     interval = INTERVAL(interval).name
    #     valid_intervals = {
    #         "m1": "1m",
    #         "m5": "5m",
    #         "m15": "15m",
    #         "m30": "30m",
    #         "h1": "1h",
    #         "d1": "1d",
    #         "w1": "1wk",
    #         "mo1": "1mo",
    #         "y1": "1y",
    #     }

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
    ) -> Tuple[Dict | None, List[DownloadRequest] | None]:
        table_name: str = f"prices_{ticker.lower()}_{VENDOR(vendor).name.lower()}_{EXCHANGE(exchange).name.lower()}_{INTERVAL(interval).name.lower()}"
        data = pd.DataFrame()
        try:
            data: pd.DataFrame = pd.read_sql_query(
                sql=f"""
                    SELECT * FROM "{table_name}" 
                    WHERE 
                        "datetime" >= '{start_datetime.strftime("%Y-%m-%d %H:%M:%S")}' 
                        AND 
                        "datetime" <= '{end_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
                """,
                con=engine,
            )
            if data.empty:
                raise ValueError
            else:
                data = CommonTasks.process_OHLC_dataframe(data)
                return (
                    CommonTasks.convert_to_json_serializable(data),
                    CommonTasks.check_missing_data(
                        ticker, data, start_datetime, end_datetime
                    ),
                )
        except:
            return (None, [DownloadRequest(ticker, start_datetime, end_datetime)])

    @staticmethod
    @celery.task
    def cache_data_to_db(
        data: Dict,
        ticker: str,
        vendor: str,
        vendor_ticker: str,
        sector: str,
        exchange: str,
        interval: int,
        instrument: str,
    ) -> None:
        table_name: str = f"prices_{ticker.lower()}_{VENDOR(vendor).name.lower()}_{EXCHANGE(exchange).name.lower()}_{INTERVAL(interval).name.lower()}"
        CommonTasks.convert_to_dataframe(data).to_sql(
            name=table_name, con=engine, if_exists="replace", index=True
        )
        try:
            Tasks.add_row(
                table_name="symbol",
                row_data={
                    "ticker": ticker,
                    "vendor_ticker": vendor_ticker,
                    "exchange": exchange,
                    "vendor": vendor,
                    "instrument": INSTRUMENT(instrument).name,
                    "name": ticker,
                    "sector": sector,
                    "interval": interval,
                    "linked_table_name": table_name,
                    "created_datetime": (datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_updated_datetime": (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                },
            )
        except:
            Tasks.edit_row(
                table_name="symbol",
                old_row_data=Tasks.get_row(
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
                    "vendor_ticker": vendor_ticker,
                    "exchange": exchange,
                    "vendor": vendor,
                    "instrument": INSTRUMENT(instrument).name,
                    "name": ticker,
                    "sector": sector,
                    "interval": interval,
                    "linked_table_name": table_name,
                    "created_datetime": (datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_updated_datetime": (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                },
            )


if __name__ == "__main__":
    celery.start()
