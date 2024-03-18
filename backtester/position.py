from datetime import datetime
from common.enums import ORDER
from backtester.order import Order


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
    # used to track the margin being used
    __margin_utilized: float

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

    @property
    def margin_utilized(self) -> float:
        return self.__margin_utilized

    @margin_utilized.setter
    def margin_utilized(self, margin_utilized: float) -> None:
        self.__margin_utilized = margin_utilized

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
        self.__margin_utilized = order.margin_utilized
