import pandas as pd
import numpy as np

from datetime import timedelta
from datetime import datetime
from Commons.common_types import *
from typing import List, Tuple, Dict, Union


class CommonTasks:
    @staticmethod
    def check_missing_data(
        ticker: str,
        dataframe: pd.DataFrame,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> List[DownloadRequest] | None:
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

        dataframe_start_datetime: datetime = dataframe.index[0].to_pydatetime()
        dataframe_end_datetime: datetime = dataframe.index[-1].to_pydatetime()

        data_appended: bool = False
        download_requests: List[DownloadRequest] = []

        if np.abs((dataframe_start_datetime - start_datetime).days) > 5:
            download_requests.append(
                DownloadRequest(
                    ticker, start_datetime - timedelta(days=1), dataframe_start_datetime
                )
            )

        if np.abs((dataframe_end_datetime - end_datetime).days) > 5:
            download_requests.append(
                DownloadRequest(
                    ticker, dataframe_end_datetime - timedelta(days=1), end_datetime
                )
            )

        return download_requests if len(download_requests) != 0 else None

    @staticmethod
    def __search(search_list: list, columns: pd.Index) -> Union[str, None]:
        for element in search_list:
            if element in columns:
                return element
        return None

    @staticmethod
    def process_OHLC_dataframe(
        dataframe: pd.DataFrame,
        datetime_index=True,
        replace_close=False,
        capital_col_names=True,
    ) -> pd.DataFrame:
        df = dataframe.copy(deep=True)
        if datetime_index:
            if df.index.name in ["Date", "date", "Datetime", "datetime"]:
                if capital_col_names:
                    df.index.name = "Datetime"
                else:
                    df.index.name = "datetime"
            elif df.index.name == None:
                col_name = CommonTasks.__search(
                    ["Date", "date", "Datetime", "datetime"], df.columns
                )
                df.set_index(col_name, inplace=True)
                df.index.name = "datetime"

            if type(df.index[0]) == str:
                df.index = pd.to_datetime(df.index)
        else:
            if df.index.name in ["Date", "date", "Datetime", "datetime"]:
                df = df.reset_index(drop=False)

        cap_names = True if "High" in df.columns else False

        if cap_names and capital_col_names:
            if replace_close and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"].values
        elif cap_names and not capital_col_names:
            if replace_close and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"].values
                df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
        elif not cap_names and capital_col_names:
            if replace_close and "adj close" in df.columns:
                df["close"] = df["adj close"].values
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "adj close": "Adj Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
        else:
            if replace_close and "adj Close" in df.columns:
                df["close"] = df["adj close"].values
        return df

    @staticmethod
    def convert_to_json_serializable(data: pd.DataFrame) -> Dict:
        if data.empty:
            return {}
        table = data.copy(deep=True)
        if pd.api.types.is_datetime64_any_dtype(table.index.to_series()):
            table.index = table.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        for column in table.columns:
            if pd.api.types.is_datetime64_any_dtype(table[column]):
                table[column] = table[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        table = table.reset_index(drop=False).to_dict(orient="records")
        return table

    @staticmethod
    def convert_to_dataframe(data: Dict) -> pd.DataFrame:
        if (len(data)) == 0:
            return pd.DataFrame()
        data: pd.DataFrame = pd.DataFrame(data)
        return CommonTasks.process_OHLC_dataframe(data)
