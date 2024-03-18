from abc import ABC, abstractmethod
from backtester.order import Order
from backtester.position import Position
from backtester.commission import Commission


class Broker(ABC):
    _cash: float
    _leverage: float
    _commission: Commission

    def __init__(self, cash: float, leverage: float, commission: Commission) -> None:
        self._cash = cash
        self._leverage = leverage
        self._commission = commission

    @property
    @abstractmethod
    def cash(self) -> float:
        pass

    @property
    @abstractmethod
    def leverage(self) -> float:
        pass

    @abstractmethod
    def commission(self) -> float:
        pass

    @property
    @abstractmethod
    def margin(self) -> float:
        pass

    @abstractmethod
    def place_order(self, order: Order) -> int:
        """Perform any order validation and return order_id if enqueued"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """Perform any order validation and return order_id if enqueued"""
        pass

    @abstractmethod
    def order_cost(self, order: Order) -> float:
        """Calculate the total value of the order and commission"""
        pass

    @abstractmethod
    def close_position(self, position_id: str) -> bool:
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
