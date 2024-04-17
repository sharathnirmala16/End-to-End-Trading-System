import psutil
import numpy as np
import pandas as pd

from typing import Type
from itertools import product
from eventus.strategy import Strategy
from eventus.commissions import Commission
from eventus.executors import BacktestExecutor
from tqdm.contrib.concurrent import process_map


def run(
    kv_pairs: dict,
) -> dict:
    bt = BacktestExecutor(**kv_pairs)

    bt.run(progress=False)
    res = bt.results()
    kpis = res["KPIs"]

    kv_pairs.pop("strategy")
    kv_pairs.pop("datetime_index")
    kv_pairs.pop("data_dict")
    kv_pairs.pop("cash")
    kv_pairs.pop("leverage")
    kv_pairs.pop("commission_model")
    kv_pairs.pop("offset")

    return {
        **kv_pairs,
        "CAGR (Ann.) [%]": kpis.loc["CAGR (Ann.) [%]"].iloc[0],
        "Volatility (Ann.) [%]": kpis.loc["Volatility (Ann.) [%]"].iloc[0],
        "Portfolio Sharpe Ratio": kpis.loc["Portfolio Sharpe Ratio"].iloc[0],
        "Max Drawdown [%]": kpis.loc["Max Drawdown [%]"].iloc[0],
        "Win Rate [%]": kpis.loc["Win Rate [%]"].iloc[0],
        "Expected Value [%]": kpis.loc["Expected Value [%]"].iloc[0],
    }


def optimize(
    strategy: Type[Strategy],
    datetime_index: np.ndarray[np.float64],
    data_dict: dict[str, np.ndarray[np.float64]],
    cash: float,
    leverage: float,
    commission_model: Commission,
    offset: int,
    params: dict[str, list],
) -> pd.DataFrame:
    params_to_send = {
        "strategy": [strategy],
        "datetime_index": [datetime_index],
        "data_dict": [data_dict],
        "cash": [cash],
        "leverage": [leverage],
        "commission_model": [commission_model],
        "offset": [offset],
        **params,
    }
    combinations = product(*(values for values in params_to_send.values()))
    kv_pairs_list = [
        {key: value for key, value in zip(params_to_send.keys(), combination)}
        for combination in combinations
    ]

    optimization_results = process_map(
        run, kv_pairs_list, max_workers=psutil.cpu_count(logical=False)
    )
    optimization_df = pd.DataFrame(optimization_results)
    results = optimization_df.set_index(list(params.keys()), inplace=False)
    return results
