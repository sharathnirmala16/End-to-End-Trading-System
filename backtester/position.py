from datetime import datetime
from common.enums import ORDER
from backtester.order import Order


class Position:
    # __position_count is used to maintain unique position_ids
    __position_count: int = 20000000
    __position_id: int
    symbol: str
    order_type: ORDER
    size: float
    price: float
    sl: float = -1
    tp: float = -1
    placed: datetime
    commission: float
    # Can be used to pass additional data to Strategy Executor based on platform
    params: dict | None
    # used to track the margin being used
    margin_utilized: float

    @property
    def position_id(self) -> int:
        return self.__position_id

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
        self.symbol = order.symbol
        self.order_type = order.order_type
        self.size = order.size
        self.price = price
        self.sl = order.sl
        self.tp = order.tp
        self.placed = placed
        self.commission = commission
        self.params = order.params
        self.margin_utilized = order.margin_utilized
