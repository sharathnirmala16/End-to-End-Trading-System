from numba import types
from numba.typed.typeddict import Dict

VENDOR = Dict.empty(key_type=types.string, value_type=types.string)
VENDOR["YAHOO"] = "Yahoo Finance"
VENDOR["BREEZE"] = "ICICI Direct Breeze API"
VENDOR["ANGELBROKING"] = "Angel Broking Smart API"


EXCHANGE = Dict.empty(key_type=types.string, value_type=types.string)
EXCHANGE["NSE"] = "National Stock Exchange"
EXCHANGE["BSE"] = "Bombay Stock Exchange"


INSTRUMENT = Dict.empty(key_type=types.string, value_type=types.string)
INSTRUMENT["STOCK"] = "Stock"
INSTRUMENT["ETF"] = "Exchange Traded Fund"
INSTRUMENT["MF"] = "Mutual Fund"
INSTRUMENT["FUTURE"] = "Future"
INSTRUMENT["OPTION"] = "Option"
INSTRUMENT["FOREX"] = "Foreign Exchange"
INSTRUMENT["CRYPTO"] = "Cryptocurrency"


INTERVAL = Dict.empty(key_type=types.string, value_type=types.int64)
INTERVAL["ms1"] = 1
INTERVAL["ms5"] = 5
INTERVAL["ms10"] = 10
INTERVAL["ms100"] = 100
INTERVAL["ms500"] = 500
INTERVAL["s1"] = 1000
INTERVAL["s5"] = 5000
INTERVAL["s15"] = 15000
INTERVAL["s30"] = 30000
INTERVAL["m1"] = 60000
INTERVAL["m5"] = 300000
INTERVAL["m15"] = 900000
INTERVAL["m30"] = 1800000
INTERVAL["h1"] = 3600000
INTERVAL["h4"] = 14400000
INTERVAL["d1"] = 86400000
INTERVAL["w1"] = 604800000
INTERVAL["mo1"] = 2592000000
INTERVAL["y1"] = 31104000000


ORDER = Dict.empty(key_type=types.string, value_type=types.int64)
ORDER["BUY"] = 1
ORDER["SELL"] = 2
ORDER["BUY_LIMIT"] = 3
ORDER["SELL_LIMIT"] = 4
