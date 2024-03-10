from abc import ABC, abstractmethod


class Commission(ABC):
    @abstractmethod
    def calculate_commission(self, price: float, size: float) -> float:
        pass


class NoCommission(Commission):
    def calculate_commission(self, price: float, size: float) -> float:
        return 0


class FlatCommission(Commission):
    amt: float

    def __init__(self, amt: float) -> None:
        self.amt = amt

    def calculate_commission(self, price: float, size: float) -> float:
        return self.amt


class PctCommission(Commission):
    """pct should be between 0 to 1"""

    pct: float

    def __init__(self, pct: float) -> None:
        if pct < 0 or pct > 1:
            raise ValueError(f"Invalid pct: 0 <= {pct} <= 1")
        self.pct = pct

    def calculate_commission(self, price: float, size: float) -> float:
        return price * abs(size) * self.pct


class PctFlatCommission(PctCommission, FlatCommission):
    """pct should be between 0 to 1"""

    def __init__(self, pct: float, amt: float) -> None:
        PctCommission.__init__(self, pct)
        FlatCommission.__init__(self, amt)

    def calculate_commission(self, price: float, size: float) -> float:
        return min(
            PctCommission.calculate_commission(self, price, size),
            FlatCommission.calculate_commission(self, price, size),
        )
