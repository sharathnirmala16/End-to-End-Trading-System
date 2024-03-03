import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchanges.nse import Nse
from vendors.breeze import Breeze
from credentials import breeze_credentials
from common.enums import INTERVAL

nse = Nse()
breeze = Breeze(breeze_credentials)

data = breeze.get_data(
    interval=INTERVAL.m5,
    exchange=nse,
    start_datetime=(datetime.today() - timedelta(days=15)),
    end_datetime=datetime.today(),
    index="NIFTY50",
)

print(data)
print(len(data))
