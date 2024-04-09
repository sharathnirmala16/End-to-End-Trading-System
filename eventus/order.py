import cython
import numpy as np

from common.typed_dicts import ORDER_TYPES
from common.exceptions import OrderError

# spec = {
#     "order_id": types.int64,
#     "symbol": types.string,
#     "order_type": types.string,  # Use String type for order_type
#     "size": types.double,
#     "price": types.double,  # Allow price to be None
#     "sl": types.double,  # Allow sl and tp to be None
#     "tp": types.double,
#     "placed": types.int64,  # Posix timestamp for placed datetime
#     "margin_utilized": types.double,
# }


# @jitclass(spec)


@cython.annotation_typing(True)
@cython.cclass
class Order:
    """
    placed is the datetime of order converted to posix time
    NOTE: FOR STOP_LOSS and TAKE_PROFIT orders, placed will store
    the position_id of the position to be closed used to track
    the margin being used
    """

    ORDER: ORDER_TYPES = ORDER_TYPES.create()
    order_count: int = 10000000
    order_id: int
    symbol: str
    order_type: str
    size: float
    price: float
    sl: float
    tp: float
    placed: int
    margin_utilized: float

    def __init__(
        self,
        symbol: str,
        order_type: str,
        placed: int,
        size: float = 1,
        price: float = np.nan,
        sl: float = np.nan,
        tp: float = np.nan,
    ) -> None:
        if size <= 0:
            raise OrderError(f"Order size <= 0")

        if order_type not in Order.ORDER:
            raise OrderError(f"{order_type} not in {Order.ORDER.keys()}")

        if order_type == "BUY":
            if sl is not np.nan and tp is not np.nan and sl > tp:
                raise OrderError(
                    f"Failed condition sl={sl} < tp={tp} for order type {order_type}"
                )
        elif order_type == "SELL":
            if sl is not np.nan and tp is not np.nan and sl < tp:
                raise OrderError(
                    f"Failed condition sl={sl} > tp={tp} for order type {order_type}"
                )

        if order_type == "BUY_LIMIT":
            if price is np.nan:
                raise OrderError(f"{order_type} must have a price")

            if sl is np.nan and tp is not np.nan and price > tp:
                raise OrderError(
                    f"Failed condition price={price} < tp={tp} for order type {order_type}"
                )
            elif sl is not np.nan and tp is np.nan and sl > price:
                raise OrderError(
                    f"Failed condition sl={sl} < price={price} for order type {order_type}"
                )
            elif sl is not np.nan and tp is not np.nan and not (sl < price < tp):
                raise OrderError(
                    f"Failed condition sl={sl} < price={price} < tp={tp} for order type {order_type}"
                )

        elif order_type == "SELL_LIMIT":
            if price is np.nan:
                raise OrderError(f"{order_type} must have a price")

            if sl is np.nan and tp is not np.nan and price < tp:
                raise OrderError(
                    f"Failed condition price={price} > tp={tp} for order type {order_type}"
                )
            elif sl is not np.nan and tp is np.nan and sl < price:
                raise OrderError(
                    f"Failed condition sl={sl} > price={price} for order type {order_type}"
                )
            elif sl is not np.nan and tp is not np.nan and not (sl > price > tp):
                raise OrderError(
                    f"Failed condition sl={sl} > price={price} > tp={tp} for order type {order_type}"
                )

        elif order_type == "STOP_LOSS":
            if price is np.nan:
                raise OrderError(f"{order_type} must have a price")

        elif order_type == "TAKE_PROFIT":
            if price is np.nan:
                raise OrderError(f"{order_type} must have a price")

        Order.order_count += 1
        self.order_id = Order.order_count
        self.symbol = symbol
        self.order_type = order_type
        self.size = size
        self.price = price
        self.sl = sl
        self.tp = tp
        self.placed = placed
        self.margin_utilized = 0
