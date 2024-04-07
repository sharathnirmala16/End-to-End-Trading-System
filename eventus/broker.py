import cython
import numpy as np

from abc import ABC, abstractmethod
from common.exceptions import TradingError
from eventus.order import Order
from eventus.position import Position
from eventus.trade import Trade
from eventus.datafeeds import DataFeed
from eventus.commissions import Commission


@cython.annotation_typing(True)
@cython.cclass
class Broker(ABC):
    cash: float
    leverage: float
    commission_model: Commission

    def __init__(
        self, cash: float, leverage: float, commission_model: Commission
    ) -> None:
        self.cash = cash
        self.leverage = leverage
        self.commission_model = commission_model

    @abstractmethod
    def place_order(self) -> int:
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        pass

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        pass

    @abstractmethod
    def close_position(self, position_id: int) -> bool:
        pass


@cython.annotation_typing(True)
@cython.cclass
class Backtester(Broker):
    """
    leverage should be less than or equal to one,
    if you want to use 10:1 leverage, enter leverage=0.1
    """

    idx: int
    orders: dict[str, dict[int, Order]]
    positions: dict[str, dict[int, Position]]
    trades: list[Trade]
    datafeed: DataFeed

    def __init__(
        self,
        cash: float,
        leverage: float,
        commission_model: Commission,
        datafeed: DataFeed,
    ) -> None:
        super.__init__(cash, leverage, commission_model)
        self.idx = 0
        self.orders = {symbol: {} for symbol in datafeed.symbols}
        self.positions = {symbol: {} for symbol in datafeed.symbols}
        self.trades = []
        self.datafeed = datafeed
        self.leverage_multiplier = 1 / self.leverage

    def commission(self, order: Order) -> float:
        return self.commission_model.calculate_commission(
            (
                self.datafeed.get_prices(order.symbol)[0]
                if order.price == np.nan
                else order.price
            ),
            order.size,
        )

    def place_order(self, order: Order) -> int:
        price = self.datafeed.get_prices(order.symbol)[0]
        comm = self.commission(order)
        cost_no_comm = order.size * price

        if self.margin > comm + cost_no_comm:
            cash_required = cost_no_comm * self.leverage
            self.cash -= cash_required
            order.margin_utilized = cost_no_comm - cash_required
            self.orders[order.symbol][order.order_id] = order
            return order.order_id
        else:
            raise TradingError(
                f"Available margin: {self.margin} is not enough, order cost is {cost_no_comm + comm}"
            )

    def __cancel_order(self, symbol: str, order_id: int) -> bool:
        order = self.orders[symbol].pop(order_id)
        self.cash += (order.size * order.price) - order.margin_utilized
        return True

    def cancel_order(self, symbol: str, order_id: int = np.nan) -> bool:
        """if order_id is np.nan, all orders of the symbol are cancelled"""
        if symbol in self.orders:
            if order_id in self.orders[symbol]:
                return self.__cancel_order(symbol, order_id)
            elif order_id is np.nan:
                for o_id in self.orders[symbol]:
                    self.__cancel_order(symbol, o_id)
                return True
            else:
                return False
        else:
            return False

    def cancel_all_orders(self) -> bool:
        for symbol in self.orders:
            for order_id in self.orders[symbol]:
                self.__cancel_order(symbol, order_id)
        return True

    @property
    def margin(self) -> float:
        return self.cash * self.leverage_multiplier
