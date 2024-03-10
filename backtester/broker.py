from abc import ABC, abstractmethod
from backtester.order import Order
from backtester.position import Position
from backtester.commission import Commission


class Broker(ABC):
    cash: float
    leverage: float
    leverage_multiplier: float
    commission: Commission

    @abstractmethod
    def bid_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def ask_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def spot_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """Perform any order validation and return true if order is enqueued"""
        pass

    @abstractmethod
    def order_cost(self, order: Order) -> float:
        """Calculate the total value of the order and commission"""
        pass

    @abstractmethod
    def close_position(self, position_id: str) -> None:
        """Closes active position with given id"""
        pass

    @property
    @abstractmethod
    def orders(self) -> list[Order]:
        """Return orders that have not been filled yet"""
        pass

    @property
    @abstractmethod
    def positions(self) -> list[Position]:
        """Return all the valid positions that are open"""
        pass
