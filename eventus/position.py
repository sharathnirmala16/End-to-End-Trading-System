from eventus.order import Order
from numba import types
from numba.experimental import jitclass
from common.exceptions import PositionError

spec = {
    "position_id": types.int64,
    "symbol": types.string,
    "order_type": types.string,  # Use String type for order_type
    "size": types.double,
    "price": types.double,  # Allow price to be None
    "sl": types.double,  # Allow sl and tp to be None
    "tp": types.double,
    "placed": types.int64,  # Posix timestamp for placed datetime
    "margin_utilized": types.double,
}


@jitclass(spec)
class Position:
    position_id: int
    symbol: str
    order_type: str
    size: float
    price: float
    sl: float
    tp: float
    # placed is the datetime of order converted to posix time
    placed: int
    commission: float
    # used to track the margin being used
    margin_utilized: float

    def __init__(
        self, order: Order, price: float, placed: int, commission: float = 0
    ) -> None:
        if order is None:
            raise PositionError("Order must be passed to create position")
        if price is None:
            raise PositionError("Position price can't be empty")
        if placed is None:
            raise PositionError("Position placement date can't be empty")

        self.position_id = 0
        self.symbol = order.symbol
        self.order_type = order.order_type
        self.size = order.size
        self.price = price
        self.sl = order.sl
        self.tp = order.tp
        self.placed = placed
        self.commission = commission
        self.margin_utilized = order.margin_utilized
