import importlib
import numpy as np
import pandas as pd

from datetime import timedelta
from vendors.yahoo import Yahoo
from exchanges.exchange import Exchange
from common.enums import INTERVAL as INTERVAL_ENUM


class PricesTransformer:
    raw_data: dict[str, pd.DataFrame]

    def __init__(
        self, raw_data: dict[str, pd.DataFrame], exchange: str, interval: str
    ) -> None:
        self.raw_data = raw_data
        self.__yahoo = Yahoo({})
        self.__interval = interval
        self.__enum_interval: INTERVAL_ENUM = getattr(INTERVAL_ENUM, interval)
        self.__exchange_obj: Exchange = getattr(
            importlib.import_module(
                name=f"exchanges.{exchange.lower()}"
            ),  # module name
            f"{exchange[0]}{exchange[1:].lower()}",  # class name,
        )()

    @staticmethod
    def balance_dataframes(
        results: dict[str, pd.DataFrame],
        interval: INTERVAL_ENUM,
        interpolation_limit: int = 2,
        filter_time: bool = False,
        filter_start_time="9:15",
        filter_end_time="15:30",
        index_combination="intersection",
    ) -> dict[str, pd.DataFrame]:
        combined_index = results[list(results.keys())[0]].index
        if index_combination == "union":
            for symbol in results:
                results[symbol] = results[symbol][~results[symbol].index.duplicated()]
                combined_index = pd.Index.union(combined_index, results[symbol].index)
            combined_index = combined_index.drop_duplicates()
        elif index_combination == "intersection":
            for symbol in results:
                results[symbol] = results[symbol][~results[symbol].index.duplicated()]
                combined_index = pd.Index.intersection(
                    combined_index, results[symbol].index
                )
            combined_index = combined_index.drop_duplicates()
        else:
            raise ValueError(
                f"index_combination={index_combination} is unknown, only 'union' and 'intersection' are supported"
            )

        for symbol in results:
            results[symbol] = (
                results[symbol]
                .apply(pd.to_numeric, errors="coerce")
                .reindex(combined_index, fill_value=np.nan)
                .interpolate(limit_direction="both", limit=interpolation_limit)
            )

        if filter_time and interval != INTERVAL_ENUM.d1:
            for symbol in results:
                results[symbol] = results[symbol].between_time(
                    pd.Timestamp(filter_start_time).time(),
                    pd.Timestamp(filter_end_time).time(),
                )
        return results

    def drop_adj_close(self) -> None:
        for symbol in self.raw_data:
            if "Adj Close" in self.raw_data[symbol].columns:
                self.raw_data[symbol] = self.raw_data[symbol].drop("Adj Close")

    def adj_internal_data(self, dataframe: pd.DataFrame) -> None:
        df = dataframe.copy(deep=True)
        adj_factor = df["Adj Close"] / df["Close"]
        df["Open"] = adj_factor * df["Open"]
        df["High"] = adj_factor * df["High"]
        df["Low"] = adj_factor * df["Low"]
        df["Close"] = df["Adj Close"]
        df["Volume"] = df["Volume"] / adj_factor
        return df

    def adj_external_data(self, dataframe: pd.DataFrame, symbol: str) -> None:
        df = dataframe.copy(deep=True)
        min_date = df.index.min().to_pydatetime()
        max_date = (df.index.max() + timedelta(days=1)).to_pydatetime()
        yf_df = self.__yahoo.get_data(
            INTERVAL_ENUM.d1,
            self.__exchange_obj,
            min_date,
            max_date,
            symbols=[symbol],
        )[symbol]
        adj_factor = yf_df["Adj Close"] / yf_df["Close"]

        resampling_interval = ""
        if 1 <= self.__enum_interval.value <= 500:
            resampling_interval = f"{self.__enum_interval.value}ms"
        elif 1000 <= self.__enum_interval.value <= 300000:
            resampling_interval = f"{self.__enum_interval.value // 1000}S"
        elif 60000 <= self.__enum_interval.value <= 1800000:
            resampling_interval = f"{self.__enum_interval.value // 60000}T"
        elif 3600000 <= self.__enum_interval.value <= 14400000:
            resampling_interval = f"{self.__enum_interval.value // 3600000}H"

        resampled_adj_factor = adj_factor.resample(resampling_interval).ffill()

        df["Open"] = resampled_adj_factor * df["Open"]
        df["High"] = resampled_adj_factor * df["High"]
        df["Low"] = resampled_adj_factor * df["Low"]
        df["Close"] = resampled_adj_factor * df["Close"]
        df["Volume"] = df["Volume"] / resampled_adj_factor
        return df.dropna()

    def adjust_prices(self) -> None:
        for symbol in self.raw_data:
            if "Adj Close" in self.raw_data[symbol].columns:
                self.raw_data[symbol] = self.adj_internal_data(self.raw_data[symbol])
            else:
                self.raw_data[symbol] = self.adj_external_data(
                    self.raw_data[symbol], symbol
                )
        self.raw_data = PricesTransformer.balance_dataframes(
            self.raw_data, self.__enum_interval
        )

    @property
    def as_pd_data(self) -> dict[str, pd.DataFrame]:
        return self.raw_data

    @property
    def as_np_data(self) -> dict[str, np.ndarray]:
        res: dict[str, np.ndarray] = {}

        for symbol in self.raw_data:
            res[symbol] = self.raw_data[symbol].values

        return res

    @property
    def dt_index(self) -> np.ndarray[np.float64]:
        symbols = list(self.raw_data.keys())
        return self.raw_data[symbols[0]].index.values.astype(np.float64)
