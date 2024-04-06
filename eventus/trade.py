from eventus.position import Position
from numba import types
from numba.typed.typeddict import Dict
from numba.experimental import jitclass
from common.exceptions import TradeError

spec = [
    ("symbol", types.string),
    ("order_type", types.string),  # Use String type for order_type
    ("size", types.double),
    ("opening_price", types.double),  # Allow price to be None
    ("closing_price", types.double),  # Allow sl and tp to be None
    ("opening_datetime", types.int64),
    ("closing_datetime", types.int64),  # Posix timestamp for placed datetime
    ("commission", types.double),
]


@jitclass(spec)
class Trade:
    symbol: str
    order_type: str
    size: float
    opening_price: float
    closing_price: float
    opening_datetime: int
    closing_datetime: int
    commission: float

    def __init__(
        self,
        open_position: Position,
        closing_price: float,
        closing_datetime: int,
        closing_commission: float = 0,
    ) -> None:
        if open_position is None:
            raise TradeError("Current position is needed for recording a trade")
        if closing_price is None:
            raise TradeError("Closing price required for recording a trade")
        if closing_datetime is None:
            raise TradeError("Closing datetime required for recording a trade")

        self.symbol = open_position.symbol
        self.order_type = open_position.order_type
        self.size = open_position.size
        self.opening_price = open_position.price
        self.closing_price = closing_price
        self.opening_datetime = open_position.placed
        self.closing_datetime = closing_datetime
        self.commission = open_position.commission + closing_commission
