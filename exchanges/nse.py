import pandas as pd

from requests import Session
from exchanges.exchange import Exchange
from io import StringIO
from enum import Enum


class NSE_INDEX(Enum):
    ALL_TICKERS = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    NIFTY50 = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    NIFTY100 = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
    NIFTY500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    NIFTYMIDCAP150 = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv"
    )
    NIFTYSMALLCAP250 = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
    )
    NIFTYMICROCAP250 = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv"
    )
    NIFTYNEXT50 = (
        "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv"
    )
    NIFTYBANK = "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv"
    NIFTYIT = "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv"
    NIFTYHEALTHCARE = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftyhealthcarelist.csv"
    )
    NIFTYFINSERVICE = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv"
    )
    NIFTYAUTO = "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv"
    NIFTYPHARMA = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv"
    )
    NIFTYFMCG = "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv"
    NIFTYMEDIA = "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv"
    NIFTYMETAL = "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv"
    NIFTYREALTY = (
        "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv"
    )


class Nse(Exchange):
    __abbreviation = "NSE"

    def __init__(self) -> None:
        super().__init__()

    @property
    def abbreviation(self) -> str:
        return self.__abbreviation

    def __init__(self) -> None:
        self.__session = Session()
        # Emulate browser
        self.__session.headers.update(
            {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"
            }
        )

        # Get the cookies from the main page (will update automatically in headers)
        self.__session.get("https://www.nseindia.com/")

    def get_symbols(self, index: str) -> dict[str, str]:
        try:
            index: NSE_INDEX = NSE_INDEX.__members__[index]
        except KeyError:
            raise ValueError(f"No member named '{index}' in {NSE_INDEX.__name__}")

        data = self.__session.get(index.value).text
        data = StringIO(data)
        df = pd.read_csv(data, sep=",")

        tickers_sector: dict[str, str] = {}
        for i in range(df.shape[0]):
            tickers_sector[df["Symbol"][i]] = df["Industry"][i]
        return tickers_sector

    def get_symbols_detailed(self, index: str) -> pd.DataFrame:
        try:
            index: NSE_INDEX = NSE_INDEX.__members__[index]
        except KeyError:
            raise ValueError(f"No member named '{index}' in {NSE_INDEX.__name__}")

        data = self.__session.get(index.value).text
        data = StringIO(data)
        df = pd.read_csv(data, sep=",")
        df.columns = df.columns.str.strip()
        return df
