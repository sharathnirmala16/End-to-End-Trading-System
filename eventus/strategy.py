import cython
import numpy as np

from abc import ABC, abstractmethod
from eventus.order import Order
from eventus.brokers import Broker
from eventus.datafeeds import DataFeed
from eventus.indicators import Indicator


@cython.annotation_typing(True)
@cython.cclass
class Strategy(ABC):
    """
    All indicators that are being used and computed should be added
    to the indicators dict for them to work properly
    """

    idx: int
    broker: Broker
    datafeed: DataFeed
    indicators: dict[str, Indicator]

    def __init__(self, broker: Broker, datafeed: DataFeed) -> None:
        self.idx = 0
        self.broker = broker
        self.datafeed = datafeed
        self.indicators = {}

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

    def cancel_order(self, symbol: str, order_id: int = np.nan) -> bool:
        return self.broker.cancel_order(symbol, order_id)

    def close_position(self, symbol: str, position_id: int = np.nan) -> bool:
        return self.broker.close_position(symbol, position_id)

    def synchronize_indexes(self) -> None:
        self.datafeed.idx = self.idx
        for indicator in self.indicators:
            self.indicators[indicator].idx = self.idx
