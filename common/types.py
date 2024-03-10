import numpy as np
import pandas as pd

from datetime import datetime
from common.enums import ORDER


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
                return sol
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


class Order:
    # __order_count is used to maintain unique order_ids
    __order_count: int = 10000000
    __order_id: int
    __symbol: str
    __order_type: ORDER
    __size: float
    __price: float
    __sl: float | None
    __tp: float | None
    __placed: datetime
    # Can be used to pass additional data to Strategy Executor based on platform
    __params: dict | None

    @property
    def order_id(self) -> int:
        return self.__order_id

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def order_type(self) -> ORDER:
        return self.__order_type

    @property
    def size(self) -> float:
        return self.__size

    @property
    def price(self) -> float:
        return self.__price

    @property
    def sl(self) -> float | None:
        return self.__sl

    @property
    def tp(self) -> float | None:
        return self.__tp

    @property
    def placed(self) -> datetime:
        return self.__placed

    @property
    def params(self) -> float | None:
        return self.__params

    def __init__(
        self,
        symbol: str,
        order_type: ORDER,
        placed: datetime,
        size: float = 1,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        **params,
    ) -> None:
        if size == 0:
            raise ValueError(f"Order size = 0")

        if order_type.name not in ORDER._member_names_:
            raise ValueError(f"{order_type.name} not in {ORDER._member_names_}")

        if order_type is ORDER.BUY:
            if sl is not None and tp is not None and sl > tp:
                raise ValueError(
                    f"Failed condition sl={sl} < tp={tp} for order type {order_type.name}"
                )
        elif order_type is ORDER.SELL:
            if sl is not None and tp is not None and sl < tp:
                raise ValueError(
                    f"Failed condition sl={sl} > tp={tp} for order type {order_type.name}"
                )

        if order_type is ORDER.BUY_LIMIT:
            if price is None:
                raise AttributeError(f"{order_type.name} must have a price")

            if sl is None and tp is not None and price > tp:
                raise ValueError(
                    f"Failed condition price={price} < tp={tp} for order type {order_type.name}"
                )
            elif sl is not None and tp is None and sl > price:
                raise ValueError(
                    f"Failed condition sl={sl} < price={price} for order type {order_type.name}"
                )
            elif sl is not None and tp is not None and not (sl < price < tp):
                raise ValueError(
                    f"Failed condition sl={sl} < price={price} < tp={tp} for order type {order_type.name}"
                )

        elif order_type is ORDER.SELL_LIMIT:
            if price is None:
                raise AttributeError(f"{order_type.name} must have a price")

            if sl is None and tp is not None and price < tp:
                raise ValueError(
                    f"Failed condition price={price} > tp={tp} for order type {order_type.name}"
                )
            elif sl is not None and tp is None and sl < price:
                raise ValueError(
                    f"Failed condition sl={sl} > price={price} for order type {order_type.name}"
                )
            elif sl is not None and tp is not None and not (sl > price > tp):
                raise ValueError(
                    f"Failed condition sl={sl} > price={price} > tp={tp} for order type {order_type.name}"
                )

        Order.__order_count += 1
        self.__order_id = Order.__order_count
        self.__symbol = symbol
        self.__order_type = order_type
        self.__size = size
        self.__price = price
        self.__sl = sl
        self.__tp = tp
        self.__placed = placed
        self.__params = params


class Position:
    # __position_count is used to maintain unique position_ids
    __position_count: int = 20000000
    __position_id: int
    __symbol: str
    __order_type: ORDER
    __size: float
    __price: float
    __sl: float | None
    __tp: float | None
    __placed: datetime
    __commission: float
    # Can be used to pass additional data to Strategy Executor based on platform
    __params: dict | None

    @property
    def position_id(self) -> int:
        return self.__position_id

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def order_type(self) -> ORDER:
        return self.__order_type

    @property
    def size(self) -> float:
        return self.__size

    @property
    def price(self) -> float:
        return self.__price

    @property
    def sl(self) -> float | None:
        return self.__sl

    @property
    def tp(self) -> float | None:
        return self.__tp

    @property
    def placed(self) -> datetime:
        return self.__placed

    @property
    def commission(self) -> float:
        return self.__commission

    @property
    def params(self) -> float | None:
        return self.__params

    def __init__(
        self, order: Order, price: float, placed: datetime, commission: float = 0
    ) -> None:
        if order is None:
            raise AttributeError("Order must be passed to create position")
        if price is None:
            raise AttributeError("Position price can't be empty")
        if placed is None:
            raise AttributeError("Position placement date can't be empty")

        Position.__position_count += 1
        self.__position_id = Position.__position_count
        self.__symbol = order.symbol
        self.__order_type = order.order_type
        self.__size = order.size
        self.__price = price
        self.__sl = order.sl
        self.__tp = order.tp
        self.__placed = placed
        self.__commission = commission
        self.__params = order.params


class Trade:
    symbol: str
    order_type: ORDER
    size: float
    opening_price: float
    closing_price: float
    opening_datetime: datetime
    closing_datetime: datetime
    commission: float

    def __init__(
        self,
        open_position: Position,
        closing_price: float,
        closing_datetime: datetime,
        closing_commission: float = 0,
    ) -> None:
        if open_position is None:
            raise AttributeError("Current position is needed for recording a trade")
        if closing_price is None:
            raise AttributeError("Closing price required for recording a trade")
        if closing_datetime is None:
            raise AttributeError("Closing datetime required for recording a trade")

        self.symbol = open_position.symbol
        self.order_type = open_position.order_type
        self.size = open_position.size
        self.opening_price = open_position.price
        self.closing_price = closing_price
        self.opening_datetime = open_position.placed
        self.closing_datetime = closing_datetime
        self.commission = open_position.commission + closing_commission

    def get_as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "order_type": self.order_type,
            "size": self.size,
            "opening_price": self.opening_price,
            "closing_price": self.closing_price,
            "opening_datetime": self.opening_datetime,
            "closing_datetime": self.closing_datetime,
            "commission": self.commission,
        }
