from datetime import datetime
from common.enums import ORDER


class Order:
    # __order_count is used to maintain unique order_ids
    __order_count: int = 10000000
    __order_id: int
    symbol: str
    order_type: ORDER
    size: float
    price: float
    sl: float | None
    tp: float | None
    placed: datetime
    # Can be used to pass additional data to Strategy Executor based on platform
    params: dict | None
    # used to track the margin being used
    margin_utilized: float

    @property
    def order_id(self) -> int:
        return self.__order_id

    def __init__(
        self,
        symbol: str,
        order_type: ORDER,
        placed: datetime,
        size: float = 1,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        **params,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Order size <= 0")

        if order_type.name not in ORDER._member_names_:
            raise ValueError(f"{order_type.name} not in {ORDER._member_names_}")

        if order_type is ORDER.BUY:
            if sl is not None and tp is not None and sl > tp:
                raise ValueError(
                    f"Failed condition sl={sl} < tp={tp} for order type {order_type.name}"
                )
        elif order_type is ORDER.SELL:
            if sl is not None and tp is not None and sl < tp:
                raise ValueError(
                    f"Failed condition sl={sl} > tp={tp} for order type {order_type.name}"
                )

        if order_type is ORDER.BUY_LIMIT:
            if price is None:
                raise AttributeError(f"{order_type.name} must have a price")

            if sl is None and tp is not None and price > tp:
                raise ValueError(
                    f"Failed condition price={price} < tp={tp} for order type {order_type.name}"
                )
            elif sl is not None and tp is None and sl > price:
                raise ValueError(
                    f"Failed condition sl={sl} < price={price} for order type {order_type.name}"
                )
            elif sl is not None and tp is not None and not (sl < price < tp):
                raise ValueError(
                    f"Failed condition sl={sl} < price={price} < tp={tp} for order type {order_type.name}"
                )

        elif order_type is ORDER.SELL_LIMIT:
            if price is None:
                raise AttributeError(f"{order_type.name} must have a price")

            if sl is None and tp is not None and price < tp:
                raise ValueError(
                    f"Failed condition price={price} > tp={tp} for order type {order_type.name}"
                )
            elif sl is not None and tp is None and sl < price:
                raise ValueError(
                    f"Failed condition sl={sl} > price={price} for order type {order_type.name}"
                )
            elif sl is not None and tp is not None and not (sl > price > tp):
                raise ValueError(
                    f"Failed condition sl={sl} > price={price} > tp={tp} for order type {order_type.name}"
                )

        Order.__order_count += 1
        self.__order_id = Order.__order_count
        self.symbol = symbol
        self.order_type = order_type
        self.size = size
        self.price = price
        self.sl = sl
        self.tp = tp
        self.placed = placed
        self.params = params
        self.margin_utilized = 0
