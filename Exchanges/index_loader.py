import numpy as np
import pandas as pd
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Dict, List
from datetime import datetime, timedelta


class IndexLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_url_dict() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def get_tickers(index: str) -> Dict[str, str]:
        pass

    @abstractproperty
    def abbreviation(self) -> str:
        pass
