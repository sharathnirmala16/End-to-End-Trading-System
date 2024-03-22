from backtester.order import Order
from backtester.position import Position
from backtester.trade import Trade
from backtester.broker import Broker
from backtester.commission import Commission
from backtester.datafeed import DataFeed
from common.exceptions import OrderError, BacktestError
from common.enums import ORDER


class BackBroker(Broker):
    __idx: int
    __orders: list[Order]
    __positions: list[Position]
    __trades: list[Trade]
    __data_feed: DataFeed
    __leverage_multiplier: float

    def __init__(
        self, cash: float, leverage: float, commission: Commission, data_feed: DataFeed
    ) -> None:
        """
        leverage should be less than or equal to one, if you want to use 10:1 leverage,
        enter leverage=0.1, __leverage_multiplier is the reciprocal of this value, i.e = 10.
        """
        super().__init__(cash, leverage, commission)
        self.__data_feed = data_feed
        self.__idx = 0
        self.__orders = []
        self.__positions = []
        self.__trades = []
        self.__leverage_multiplier = 1 / self._leverage

    def __fill_order(self, order: Order) -> None:
        comm = self._commission.calculate_commission(order.price, order.size)
        if self._cash < comm:
            raise BacktestError(
                f"Cash: {self._cash} is not enough to continue trading."
            )
        else:
            self._cash -= comm
            self.__positions.append(
                Position(
                    order=order,
                    price=self.__data_feed.spot_price(order.symbol),
                    placed=self.__data_feed.current_datetime,
                    commission=comm,
                )
            )

    def fill_orders(self) -> None:
        # commission is taken when order is filled
        if len(self.__orders) == 0:
            return None

        unfilled_orders: list[Order] = []
        for order in self.__orders:
            if (
                (order.order_type == ORDER.BUY or order.order_type == ORDER.SELL)
                or (
                    order.order_type == ORDER.BUY_LIMIT
                    and order.price <= self.__data_feed.spot_price(order.symbol)
                )
                or (
                    order.order_type == ORDER.SELL_LIMIT
                    and order.price >= self.__data_feed.spot_price(order.symbol)
                )
            ):
                self.__fill_order(order)
            else:
                unfilled_orders.append(order)
        self.__orders = unfilled_orders

    def __price_triggered(self, position: Position, current_price: float) -> bool:
        current_price = self.__data_feed.spot_price(position.symbol)
        if (
            position.order_type == ORDER.BUY
            or position.order_type == ORDER.BUY_LIMIT
            and (current_price <= self.sl or current_price >= self.tp)
        ):
            return True
        if (
            position.order_type == ORDER.SELL
            or position.order_type == ORDER.SELL_LIMIT
            and (current_price >= self.sl or current_price <= self.tp)
        ):
            return True
        return False

    def close_positions_tp_sl(self) -> None:
        if len(self.__positions) == 0:
            return None

        for index, position in enumerate(self.__positions):
            if self.__price_triggered(position):
                self.__close_position(index)

    def __close_position(self, pos_index: int) -> bool:
        position = self.__positions.pop(pos_index)
        current_price = self.__data_feed.spot_price(position.symbol)
        self.cash += (
            (position.size * current_price)
            - position.margin_utilized
            - self._commission.calculate_commission(current_price, position.size)
        )
        # Correction for short orders
        if position.order_type == ORDER.SELL or position.order_type == ORDER.SELL_LIMIT:
            self._cash += 2 * (position.price - current_price)
        return True

    def close_position(self, position_id: int) -> bool:
        pos_index = -1
        for index, position in enumerate(self.__positions):
            if position.position_id == position_id:
                pos_index = index
                break

        if pos_index == -1:
            return False
        else:
            return self.__close_position(pos_index)

    def close_all_positions(self) -> None:
        for index, _ in enumerate(self.__positions):
            self.__close_position(index)

    def order_cost(self, order: Order) -> float:
        return (order.price * order.size) + self.commission(order)

    def commission(self, order: Order) -> float:
        return self._commission.calculate_commission(
            (
                self.__data_feed.spot_price(order.symbol)
                if order.price is None
                else order.price
            ),
            order.size,
        )

    def place_order(self, order: Order) -> int:
        comm = self.commission(order)
        cost_no_comm = order.size * order.price

        if self.margin > comm + cost_no_comm:
            cash_required = cost_no_comm * self._leverage
            self._cash -= cash_required
            order.margin_utilized = cost_no_comm - cash_required
            self.__orders.append(order)
            return order.order_id
        else:
            raise OrderError(
                f"Available margin: {self._broker.margin} is not enough, order cost is {cost_no_comm + comm}"
            )

    def __cancel_order(self, order_index: int) -> bool:
        order = self.__orders.pop(order_index)
        self._cash += (order.size * order.price) - order.margin_utilized
        return True

    def cancel_order(self, order_id: int) -> bool:
        order_index = -1
        for index, order in enumerate(self.__orders):
            if order.order_id == order_id:
                order_index = index
                break

        if order_index == -1:
            return False

        else:
            return self.__cancel_order(order_index)

    def canel_all_orders(self) -> None:
        for index, _ in enumerate(self.__orders):
            self.__cancel_order(index)

    @property
    def margin(self) -> float:
        return self._cash * self.__leverage_multiplier

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def leverage(self) -> float:
        return self._leverage

    @property
    def leverage_multiplier(self) -> float:
        return self.__leverage_multiplier

    @property
    def idx(self) -> int:
        return self.__idx

    @idx.setter
    def idx(self, idx: int) -> None:
        self.__idx = idx

    @property
    def orders(self) -> list[Order]:
        return self.__orders

    @orders.setter
    def orders(self, orders: list[Order]) -> None:
        self.__orders = orders

    @property
    def positions(self) -> list[Position]:
        return self.__positions

    @property
    def trades(self) -> list[Trade]:
        return self.__trades
