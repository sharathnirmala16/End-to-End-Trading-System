from datetime import datetime
from common.enums import ORDER


class Order:
    # __order_count is used to maintain unique order_ids
    __order_count: int = 10000000
    __order_id: int
    __symbol: str
    __order_type: ORDER
    __size: float
    __price: float
    __sl: float | None
    __tp: float | None
    __placed: datetime
    # Can be used to pass additional data to Strategy Executor based on platform
    __params: dict | None

    @property
    def order_id(self) -> int:
        return self.__order_id

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def order_type(self) -> ORDER:
        return self.__order_type

    @property
    def size(self) -> float:
        return self.__size

    @property
    def price(self) -> float:
        return self.__price

    @property
    def sl(self) -> float | None:
        return self.__sl

    @property
    def tp(self) -> float | None:
        return self.__tp

    @property
    def placed(self) -> datetime:
        return self.__placed

    @property
    def params(self) -> float | None:
        return self.__params

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
        if size == 0:
            raise ValueError(f"Order size = 0")

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
        self.__symbol = symbol
        self.__order_type = order_type
        self.__size = size
        self.__price = price
        self.__sl = sl
        self.__tp = tp
        self.__placed = placed
        self.__params = params
