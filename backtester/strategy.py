from common.enums import ORDER
from common.exceptions import OrderError
from backtester.order import Order
from backtester.broker import Broker
from backtester.datafeed import DataFeed
from datetime import datetime


class Strategy:
    _broker: Broker
    _data_feed: DataFeed

    def __init__(self, broker: Broker, data_feed: DataFeed) -> None:
        self._broker = broker
        self._data_feed = data_feed

    def buy(
        self,
        symbol: str,
        order_type: ORDER,
        size: float,
        placed: datetime,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> int:
        new_order = Order(
            symbol=symbol,
            order_type=order_type,
            size=size,
            price=price,
            sl=sl,
            tp=tp,
            placed=placed,
        )
        return self._broker.place_order(new_order)

    def sell(
        self,
        symbol: str,
        order_type: ORDER,
        size: float,
        placed: datetime,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> int:
        new_order = Order(
            symbol=symbol,
            order_type=order_type,
            size=size,
            price=price,
            sl=sl,
            tp=tp,
            placed=placed,
        )
        return self._broker.place_order(new_order)

    def cancel_order(self, order_id: int) -> bool:
        return self._broker.cancel_order(order_id)

    def close_position(self, position_id: int) -> bool:
        return self._broker.close_position(position_id)

    def init(self) -> None:
        """Overload when defining your own strategy"""
        pass

    def next(self) -> None:
        """Overload when defining your own strategy"""
        pass
