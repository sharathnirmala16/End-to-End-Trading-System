from datetime import datetime
from common.enums import ORDER
from backtester.order import Order
from backtester.position import Position


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
