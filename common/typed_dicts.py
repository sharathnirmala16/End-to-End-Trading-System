from __future__ import annotations
from typing import TypedDict


class VENDOR(TypedDict):
    YAHOO: str
    BREEZE: str
    ANGELBROKING: str

    @classmethod
    def create(cls) -> VENDOR:
        return VENDOR(
            YAHOO="Yahoo Finance",
            BREEZE="ICICI Breeze",
            ANGELBROKING="Angel Broking Smart API",
        )


class EXCHANGE(TypedDict):
    NSE: str = "National Stock Exchange"
    BSE: str = "Bombay Stock Exchange"

    @classmethod
    def create(cls) -> EXCHANGE:
        return EXCHANGE(NSE="National Stock Exchange", BSE="Bombay Stock Exchange")


class INSTRUMENT(TypedDict):
    STOCK = "Stock"
    ETF = "Exchange Traded Fund"
    MF = "Mutual Fund"
    FUTURE = "Future"
    OPTION = "Option"
    FOREX = "Foreign Exchange"
    CRYPTO = "Cryptocurrency"

    @classmethod
    def create(cls) -> EXCHANGE:
        return EXCHANGE(
            STOCK="Stock",
            ETF="Exchange Traded Fund",
            MF="Mutual Fund",
            FUTURE="Future",
            OPTION="Option",
            FOREX="Foreign Exchange",
            CRYPTO="Cryptocurrency",
        )


class INTERVAL(TypedDict):
    ms1: int
    ms5: int
    ms10: int
    ms100: int
    ms500: int
    s1: int
    s5: int
    s15: int
    s30: int
    m1: int
    m5: int
    m15: int
    m30: int
    h1: int
    h4: int
    d1: int
    w1: int
    mo1: int
    y1: int

    @classmethod
    def create(cls) -> INTERVAL:
        return INTERVAL(
            ms1=1,
            ms5=5,
            ms10=10,
            ms100=100,
            ms500=500,
            s1=1000,
            s5=5000,
            s15=15000,
            s30=30000,
            m1=60000,
            m5=300000,
            m15=900000,
            m30=1800000,
            h1=3600000,
            h4=14400000,
            d1=86400000,
            w1=604800000,
            mo1=2592000000,
            y1=31104000000,
        )


class ORDER_TYPES(TypedDict):
    BUY: int
    SELL: int
    BUY_LIMIT: int
    SELL_LIMIT: int
    STOP_LOSS: int
    TAKE_PROFIT: int

    @classmethod
    def create(cls) -> ORDER_TYPES:
        return ORDER_TYPES(
            BUY=1,
            SELL=2,
            BUY_LIMIT=3,
            SELL_LIMIT=4,
            STOP_LOSS=5,
            TAKE_PROFIT=6,
        )
