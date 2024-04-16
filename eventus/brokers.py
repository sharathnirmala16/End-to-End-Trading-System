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
    orders: dict[str, dict[int, Order]]
    positions: dict[str, dict[int, Position]]
    trades: list[Trade]
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
    def cancel_order(self, symbol: str, order_id: int = np.nan) -> bool:
        pass

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        pass

    @abstractmethod
    def close_position(self, symbol: str, position_id: int = np.nan) -> bool:
        pass

    @abstractmethod
    def close_all_positions(self) -> bool:
        pass

    @abstractmethod
    def synchronize_indexes(self) -> None:
        pass

    @property
    @abstractmethod
    def open_positions_count(self) -> float:
        pass

    @property
    @abstractmethod
    def current_equity(self) -> float:
        pass

    @property
    @abstractmethod
    def margin(self) -> float:
        pass


@cython.annotation_typing(True)
@cython.cclass
class Backtester(Broker):
    """
    leverage should be less than or equal to one,
    if you want to use 10:1 leverage, enter leverage=0.1,
    for backtesting purposes, stop losses and take profits
    are also converted to limit orders and placed in the orders dict
    """

    idx: int
    datafeed: DataFeed

    def __init__(
        self,
        cash: float,
        leverage: float,
        commission_model: Commission,
        datafeed: DataFeed,
    ) -> None:
        super().__init__(cash, leverage, commission_model)
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
                if order.price is np.nan
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
        if order.order_type not in {"STOP_LOSS", "TAKE_PROFIT"}:
            self.cash += (order.size * order.price) - order.margin_utilized
        return True

    def cancel_order(self, symbol: str, order_id: int = np.nan) -> bool:
        """if order_id is np.nan, all orders of the symbol are cancelled"""
        if symbol in self.orders:
            if order_id in self.orders[symbol]:
                return self.__cancel_order(symbol, order_id)
            elif order_id is np.nan:
                order_ids = list(self.orders[symbol].keys())
                for o_id in order_ids:
                    self.__cancel_order(symbol, o_id)
                return True
            else:
                return False
        else:
            return False

    def cancel_all_orders(self) -> bool:
        for symbol in self.orders:
            orders_ids = list(self.orders[symbol].keys())
            for order_id in orders_ids:
                self.__cancel_order(symbol, order_id)
        return True

    def __fill_order(self, order: Order, price: float) -> None:
        exec_price = price if order.price is np.nan else order.price
        comm = self.commission(order)
        if self.cash < comm:
            # raise TradingError(f"Cash: {self.cash} is not enough to continue trading.")
            return
        else:
            pos = Position(
                order, exec_price, self.datafeed.get_datetime_index()[0], comm
            )
            self.cash -= comm
            self.positions[pos.symbol][pos.position_id] = pos
            self.orders[order.symbol].pop(order.order_id)
            # No commission and margin charges on the generated sl and tp orders, these are charged when these orders are filled
            if pos.sl is not np.nan:
                sl_order = Order(
                    symbol=pos.symbol,
                    order_type="STOP_LOSS",
                    placed=pos.position_id,
                    size=pos.size,
                    price=pos.sl,
                )
                self.orders[order.symbol][sl_order.order_id] = sl_order
            if pos.tp is not np.nan:
                tp_order = Order(
                    symbol=pos.symbol,
                    order_type="TAKE_PROFIT",
                    placed=pos.position_id,
                    size=pos.size,
                    price=pos.tp,
                )
                self.orders[order.symbol][tp_order.order_id] = tp_order

    def fill_orders(self) -> None:
        for symbol in self.orders:
            if len(self.orders[symbol]) != 0:
                order_ids = list(self.orders[symbol])
                for order_id in order_ids:
                    order = self.orders[symbol][order_id]
                    price = self.datafeed.get_prices(order.symbol)[0]
                    if (
                        (order.order_type in {"BUY", "SELL"})
                        or (order.order_type == "BUY_LIMIT" and order.price <= price)
                        or (order.order_type == "SELL_LIMIT" and order.price >= price)
                    ):
                        self.__fill_order(order, price)
                    elif order.order_type in {"STOP_LOSS", "TAKE_PROFIT"}:
                        self.__fill_sl_tp_orders(order)

    def __clear_sl_tp_orders(self, symbol: str, position_to_close_id: int) -> None:
        order_ids = list(self.orders[symbol].keys())
        for order_id in order_ids:
            if self.orders[symbol][order_id].placed == position_to_close_id:
                self.orders.pop(order_id)

    def __fill_sl_tp_orders(self, order: Order) -> None:
        if order.placed not in self.positions[order.symbol]:
            return
        pos_to_close = self.positions[order.symbol][order.placed]
        price = self.datafeed.get_prices(order.symbol)[0]
        if order.order_type == "STOP_LOSS":
            if (
                pos_to_close.order_type in {"BUY", "BUY_LIMIT"} and price <= order.price
            ) or (
                pos_to_close.order_type in {"SELL", "SELL_LIMIT"}
                and price >= order.price
            ):
                self.__close_position(pos_to_close.symbol, pos_to_close.position_id)
        elif order.order_type == "TAKE_PROFIT":
            if (
                pos_to_close.order_type in {"BUY", "BUY_LIMIT"} and price >= order.price
            ) or (
                pos_to_close.order_type in {"SELL", "SELL_LIMIT"}
                and price <= order.price
            ):
                self.__close_position(pos_to_close.symbol, pos_to_close.position_id)

    def __close_position(self, symbol: str, position_id: int) -> bool:
        position = self.positions[symbol].pop(position_id)
        current_price = self.datafeed.get_prices(position.symbol)[0]
        comm = self.commission_model.calculate_commission(current_price, position.size)
        self.cash += (position.size * current_price) - position.margin_utilized - comm
        if position.order_type == "SELL" or position.order_type == "SELL_LIMIT":
            self.cash += 2 * (position.price - current_price)
        if position.order_type == "STOP_LOSS" or position.order_type == "TAKE_PROFIT":
            self.__clear_sl_tp_orders(symbol, position_id)
        self.trades.append(
            Trade(position, current_price, self.datafeed.get_datetime_index()[0], comm)
        )
        return True

    def close_position(self, symbol: str, position_id: int = np.nan) -> bool:
        """if order_id is np.nan, all positions of the symbol are close"""
        if symbol in self.positions:
            if position_id in self.positions[symbol]:
                return self.__close_position(symbol, position_id)
            else:
                position_ids = list(self.positions[symbol].keys())
                for p_id in position_ids:
                    self.__close_position(symbol, p_id)
                return True
        else:
            return False

    def close_all_positions(self) -> bool:
        for symbol in self.positions:
            position_ids = list(self.positions[symbol].keys())
            for position_id in position_ids:
                self.__close_position(symbol, position_id)
        return True

    def synchronize_indexes(self) -> None:
        self.datafeed.idx = self.idx

    @property
    def margin(self) -> float:
        return self.cash * self.leverage_multiplier

    @property
    def current_equity(self) -> float:
        position_equity = 0
        for symbol in self.positions:
            for position_id in self.positions[symbol]:
                position = self.positions[symbol][position_id]
                position_equity += (
                    position.size * self.datafeed.get_prices(position.symbol)[0]
                    - position.margin_utilized
                )
        return position_equity + self.cash

    @property
    def open_positions_count(self) -> float:
        return sum(map(lambda positions: len(positions), self.positions.values()))
