import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime
from credentials import breeze_credentials, psql_credentials
from common.enums import *
from securities_master.securities_master import SecuritiesMaster

sm = SecuritiesMaster(
    vendor_login_credentials={
        VENDOR.YAHOO.name: {},
        VENDOR.BREEZE.name: breeze_credentials,
    },
    db_credentials=psql_credentials,
)

# lt_data = yf.download("LT.NS", start="2017-12-20", end="2020-01-10")
# print(sm.cache_data_to_db(lt_data, "prices_lt_yahoo_nse_d1"))

print(
    sm.get_prices(
        interval="d1",
        start_datetime=datetime(2018, 6, 1),
        end_datetime=datetime(2020, 6, 1),
        vendor="YAHOO",
        exchange="NSE",
        instrument="STOCK",
        symbols=None,
        index="NIFTY50",
        adjusted_prices=False,
        drop_adjusted_prices=False,
        cache_data=True,
    )
)
print(sm.get_table("symbol"))
# print(
#     sm.get_prices(
#         interval="d1",
#         start_datetime=datetime(2018, 1, 1),
#         end_datetime=datetime(2019, 1, 1),
#         vendor="YAHOO",
#         exchange="NSE",
#         instrument="STOCK",
#         symbols=["TCS", "LT", "RELIANCE"],
#         index=None,
#         adjusted_prices=False,
#         drop_adjusted_prices=False,
#         cache_data=False,
#     )
# )
