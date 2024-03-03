import pandas as pd

from abc import ABC, abstractmethod


class Exchange(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def abbreviation(self) -> str:
        pass

    @abstractmethod
    def get_symbols(self, index: str) -> dict[str, str]:
        pass

    @abstractmethod
    def get_symbols_detailed(self, index: str) -> pd.DataFrame:
        pass
