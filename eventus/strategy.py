import cython
import numpy as np

from abc import ABC, abstractmethod
from eventus.order import Order
from eventus.broker import Broker
from eventus.datafeeds import DataFeed


@cython.annotation_typing(True)
@cython.cclass
class Strategy(ABC):
    idx: int
    broker: Broker
    datafeed: DataFeed

    def __init__(self, broker: Broker, datafeed: DataFeed) -> None:
        self.idx = 0
        self.broker = Broker
        self.datafeed = datafeed

    def init(self):
        pass

    def next(self):
        pass

    def buy(
        self,
        symbol: str,
        limit_order: bool = False,
        size: float = 1.0,
        price: float = np.nan,
        sl: float = np.nan,
        tp: float = np.nan,
    ) -> int:
        return self.broker.place_order(
            Order(
                symbol=symbol,
                order_type="BUY_LIMIT" if limit_order else "BUY",
                placed=self.datafeed.get_datetime_index(window=1)[0],
                size=size,
                price=price,
                sl=sl,
                tp=tp,
            )
        )

    def sell(
        self,
        symbol: str,
        limit_order: bool = False,
        size: float = 1.0,
        price: float = np.nan,
        sl: float = np.nan,
        tp: float = np.nan,
    ) -> int:
        return self.broker.place_order(
            Order(
                symbol=symbol,
                order_type="SELL_LIMIT" if limit_order else "SELL",
                placed=self.datafeed.get_datetime_index(window=1)[0],
                size=size,
                price=price,
                sl=sl,
                tp=tp,
            )
        )

    def cancel_order(self, order_id: int) -> bool:
        return self.broker.cancel_order(order_id)

    def cancel_position(self, position_id: int) -> bool:
        return self.broker.close_position(position_id)
