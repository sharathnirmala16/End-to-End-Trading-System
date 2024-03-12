import numpy as np
import pandas as pd


class AssetsData:
    __data_array: np.ndarray[np.float64]
    __tickers_dict: dict[str, int]
    __cols_dict: dict[str, int]

    def __init__(self, data_dict: dict[str, pd.DataFrame]) -> None:
        self.__tickers_dict = {
            ticker: index for index, ticker in enumerate(data_dict.keys())
        }
        self.__cols_dict = {
            "Datetime": 0,
            "Open": 1,
            "High": 2,
            "Low": 3,
            "Close": 4,
            "Volume": 5,
        }
        self.__offset = len(self.__cols_dict) - 1
        # shape is rows=number of rows in the first df, cols= 5 (for OHLCV) * number of tickers + 1 (for the Datetime)
        self.__data_array = np.zeros(
            shape=(
                data_dict[self.tickers[0]].shape[0],
                self.__offset * len(self.tickers) + 1,
            )
        )

        # assiging datetime column to the zeroth column
        self.__data_array[:, 0] = data_dict[self.tickers[0]].index.values.astype(
            np.float64
        )
        start, end = 1, self.__offset + 1
        for ticker in self.tickers:
            self.__data_array[:, start:end] = data_dict[ticker].values
            start = end
            end = start + self.__offset

    def __getitem__(self, key: int | str | list[str, str]) -> np.ndarray[np.float64]:
        if isinstance(key, int):
            return self.__data_array[key]

        elif isinstance(key, str):
            if key in self.tickers:
                sol: np.ndarray[np.float64] = np.zeros(
                    shape=(self.__data_array.shape[0], self.__offset + 1)
                )
                index = self.__tickers_dict[key]
                sol[:, 0] = self.__data_array[:, 0]
                sol[:, 1 : self.__offset + 1] = self.__data_array[
                    :,
                    (self.__offset * index + 1) : (
                        self.__offset * (index + 1) + 1
                    ),  # self.__offset * index + self.__offset + 1 =  self.__offset * (index + 1) + 1
                ]
                return sol

            elif key in self.columns:
                sol: np.ndarray[np.float64] = np.zeros(
                    shape=(self.__data_array.shape[0], len(self.tickers) + 1)
                )
                sol[:, 0] = self.__data_array[:, 0]
                index = self.__cols_dict[key]
                for i in range(1, len(self.tickers) + 1):
                    sol[:, i] = self.__data_array[:, index]
                    index += self.__offset
                return sol

            else:
                raise Exception(
                    f"Invalid key, neither in {self.tickers} or {self.columns}"
                )

        elif isinstance(key, list) and len(key) == 2:
            if key[0] in self.tickers and key[1] in self.columns:
                sol: np.ndarray[np.float64] = np.zeros(
                    shape=(self.__data_array.shape[0], 2)
                )
                sol[:, 0] = self.__data_array[:, 0]
                sol[:, 1] = self.__data_array[
                    :,
                    (
                        self.__offset * self.__tickers_dict[key[0]]
                        + self.__cols_dict[key[1]]
                    ),
                ]
                return sol

            elif key[0] in self.tickers and isinstance(key[1], int):
                sol: np.ndarray[np.float64] = np.zeros(shape=(1, self.__offset + 1))
                index = self.__tickers_dict[key[0]]
                sol[0, 0] = self.__data_array[key[1], 0]
                sol[0, 1:] = self.__data_array[
                    key[1],
                    (self.__offset * index + 1) : (
                        self.__offset * index + (self.__offset * (index + 1) + 1)
                    ),
                ]
                return sol

            elif key[0] in self.columns and isinstance(key[1], int):
                sol: np.ndarray[np.float64] = np.zeros(shape=(1, len(self.tickers) + 1))
                col_offset = self.__cols_dict[key[0]]
                sol[0, 0] = self.__data_array[key[1], 0]
                for i in range(1, len(self.tickers) + 1):
                    sol[0, i] = self.__data_array[
                        key[1], (self.__offset * (i - 1) + col_offset)
                    ]
                return sol

            else:
                raise Exception(
                    f"Invalid key, either or both {key} missing in {self.tickers} and {self.columns}"
                )

        elif isinstance(key, list) and len(key) == 3:
            if (
                key[0] in self.tickers
                and key[1] in self.columns
                and isinstance(key[2], int)
            ):
                sol = np.zeros(shape=(1, 2))
                sol[0, 0] = self.__data_array[key[2], 0]
                sol[0, 1] = self.__data_array[
                    key[2],
                    self.__offset * self.__tickers_dict[key[0]]
                    + self.__cols_dict[key[1]],
                ]
                return sol[0]
            else:
                raise Exception(
                    f"Invalid key, order of your key [{key}] doesn't match pattern [['ticker', 'price', 'int']]"
                )

        else:
            raise Exception(
                """Invalid indexing, these are the possible ways: 
                ['int'], ['str'], [['ticker', 'price']], [['ticker', 'int']], 
                [['price', 'int']], [['ticker', 'price', 'int']]"""
            )

    @property
    def index(self) -> np.ndarray[np.float64]:
        return self.__data_array[:, 0]

    @property
    def data_array(self) -> np.ndarray[np.float64]:
        return self.__data_array

    @property
    def tickers(self) -> list[str]:
        return list(self.__tickers_dict.keys())

    @property
    def columns(self) -> list[str]:
        return self.__cols_dict.keys()
