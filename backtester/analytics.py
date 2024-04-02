import numpy as np
import pandas as pd

from datetime import datetime
from common.enums import ORDER
from backtester.trade import Trade


class Analyzer:
    def __init__(self, intermediate_results: dict) -> None:
        self.__results = intermediate_results
        self.compute_results()

    def compute_results(self) -> dict[str, pd.DataFrame]:
        self.__results["Trades"] = self.parse_trades()
        self.__results["Trading Duration [d]"] = self.total_trade_duration()
        self.__results["Total Returns [%]"] = (
            (self.__results["Ending Cash [$]"] / self.__results["Starting Cash [$]"])
            - 1
        ) * 100
        self.__results["CAGR (Ann.) [%]"] = self.cagr()
        self.__results["Volatility (Ann.) [%]"] = self.volatility()
        self.__results["Sharpe Ratio"] = self.sharpe()
        self.__results["Max Drawdown [%]"] = self.max_drawdown()
        self.__results["Number of Trades"] = self.__results["Trades"].shape[0]
        (
            self.__results["Win Rate [%]"],
            self.__results["Avg Win [%]"],
            self.__results["Avg Loss [%]"],
            self.__results["Best Win [%]"],
            self.__results["Worst Loss [%]"],
            self.__results["Profit Factor"],
        ) = self.win_stats()
        self.__results["Expected Value [%]"] = (
            (self.__results["Win Rate [%]"] * self.__results["Avg Win [%]"])
            + ((100 - self.__results["Win Rate [%]"]) * self.__results["Avg Loss [%]"])
        ) / 100
        (
            self.__results["Avg Trade Duration [d]"],
            self.__results["Max Trade Duration [d]"],
        ) = self.trade_duration_stats()

    def parse_trades(self) -> pd.DataFrame:
        trades_list: list[list] = [
            [
                trade.symbol,
                trade.order_type.name,
                trade.size,
                trade.opening_price,
                trade.closing_price,
                trade.opening_datetime,
                trade.closing_datetime,
                trade.commission,
            ]
            for trade in self.__results["Trades"]
        ]
        df = pd.DataFrame(
            trades_list,
            columns=[
                "Symbol",
                "Order Type",
                "Size",
                "Opening Price",
                "Closing Price",
                "Opening Datetime",
                "Closing Datetime",
                "Commission",
            ],
        )
        df["Directional Size"] = df["Size"]
        df.loc[
            df["Order Type"].isin([ORDER.SELL.name, ORDER.SELL_LIMIT.name]),
            "Directional Size",
        ] *= -1
        df["Gross PnL"] = df["Directional Size"] * (
            df["Closing Price"] - df["Opening Price"]
        )
        df["Net PnL"] = df["Gross PnL"] - df["Commission"]
        df["Net PnL [%]"] = (df["Net PnL"] / (df["Opening Price"] * df["Size"])) * 100
        return df

    def total_trade_duration(self, result_unit: str = "days") -> float:
        """result_unit: str can be any of the following: ["minutes", "hours", "days", "years"]"""
        # NOTE: Rather than trades table, maybe use equity curve's index?
        if self.__results["Equity Curve [$]"].shape[0] == 0:
            return 0
        start: datetime = self.__results["Equity Curve [$]"].index[0]
        end: datetime = self.__results["Equity Curve [$]"].index[-1]
        match result_unit:
            case "minutes":
                return (end - start).total_seconds() / 60
            case "hours":
                return (end - start).total_seconds() / (60 * 60)
            case "days":
                return (end - start).total_seconds() / (60 * 60 * 24)
            case "years":
                # 252 instead of 365 because there are 252 trading days
                return (end - start).total_seconds() / (60 * 60 * 24 * 252)

    def cagr(self) -> float:
        if self.total_trade_duration(result_unit="years") == 0:
            return np.nan
        return (
            (
                (
                    self.__results["Ending Cash [$]"]
                    / self.__results["Starting Cash [$]"]
                )
                ** (1 / self.total_trade_duration(result_unit="years"))
            )
            - 1
        ) * 100

    def volatility(self) -> float:
        return (
            self.__results["Equity Curve [$]"].pct_change().std() * np.sqrt(252) * 100
        )

    def sharpe(self) -> float:
        if self.__results["Volatility (Ann.) [%]"] == 0:
            return np.nan
        return (
            self.__results["CAGR (Ann.) [%]"] / self.__results["Volatility (Ann.) [%]"]
        )

    def win_stats(self) -> tuple[float, float, float, float, float, float]:
        trades: pd.DataFrame = self.__results["Trades"].copy(deep=True)
        win_rate = ((trades["Net PnL"] > 0).sum() / trades.shape[0]) * 100
        avg_win = trades.loc[(trades["Net PnL [%]"] > 0), "Net PnL [%]"].mean()
        avg_loss = trades.loc[(trades["Net PnL [%]"] <= 0), "Net PnL [%]"].mean()
        best_win = trades.loc[(trades["Net PnL [%]"] > 0), "Net PnL [%]"].max()
        worst_loss = trades.loc[(trades["Net PnL [%]"] <= 0), "Net PnL [%]"].min()
        profit_factor = np.nan
        if trades.loc[(trades["Net PnL"] <= 0), "Net PnL [%]"].sum() != 0:
            profit_factor = trades.loc[
                (trades["Net PnL"] > 0), "Net PnL"
            ].sum() / np.abs(trades.loc[(trades["Net PnL"] <= 0), "Net PnL"].sum())
        return win_rate, avg_win, avg_loss, best_win, worst_loss, profit_factor

    def trade_duration_stats(self) -> tuple[float, float]:
        df: pd.DataFrame = self.__results["Trades"].copy(deep=True)
        return (df["Closing Datetime"] - df["Opening Datetime"]).mean(), (
            df["Closing Datetime"] - df["Opening Datetime"]
        ).max()

    def max_drawdown(self) -> float:
        eq_curve: pd.Series = self.__results["Equity Curve [$]"].copy(deep=True)
        returns: pd.Series = eq_curve.pct_change()
        cum_returns: pd.Series = (1 + returns).cumprod()
        cum_roll_max: pd.Series = cum_returns.cummax()
        drawdown: pd.Series = cum_roll_max - cum_returns
        return (drawdown / cum_roll_max).max() * 100

    @property
    def results(self) -> dict[str, pd.DataFrame]:
        kpis = self.__results
        trades: pd.DataFrame = kpis.pop("Trades")
        eq_curve: pd.DataFrame = pd.DataFrame(
            kpis.pop("Equity Curve [$]"), columns=["Equity [$]"]
        )
        kpis: pd.DataFrame = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
        kpis = kpis.set_index("KPI")
        return {"KPIs": kpis, "Trades": trades, "Equity Curve [$]": eq_curve}
