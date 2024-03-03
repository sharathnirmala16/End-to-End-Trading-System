from enum import Enum


class VENDOR(Enum):
    YAHOO = "Yahoo Finance"
    BREEZE = "ICICI Direct Breeze API"
    ANGELBROKING = "Angel Broking Smart API"


class EXCHANGE(Enum):
    NSE = "National Stock Exchange"
    BSE = "Bombay Stock Exchange"


class INSTRUMENT(Enum):
    STOCK = "Stock"
    ETF = "Exchange Traded Fund"
    MF = "Mutual Fund"
    FUTURE = "Future"
    OPTION = "Option"
    FOREX = "Foreign Exchange"
    CRYPTO = "Cryptocurrency"


class INTERVAL(Enum):
    ms1 = 1
    ms5 = 5
    ms10 = 10
    ms100 = 100
    ms500 = 500
    s1 = 1000
    s5 = 5000
    s15 = 15000
    s30 = 30000
    m1 = 60000
    m5 = 300000
    m15 = 900000
    m30 = 1800000
    h1 = 3600000
    h4 = 14400000
    d1 = 86400000
    w1 = 604800000
    mo1 = 2592000000
    y1 = 31104000000
