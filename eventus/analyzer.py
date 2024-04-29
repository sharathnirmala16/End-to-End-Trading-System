import numpy as np
import pandas as pd

from eventus.trade import Trade
from datetime import datetime


class Analyzer:
    def __init__(self, equity_curve: list[list], trades: list[Trade]) -> None:
        self.results: dict = {}
        equity_array = np.array(equity_curve)
        self.equity_curve = pd.Series(
            equity_array[:, 1], index=pd.to_datetime(equity_array[:, 0] / 1e9, unit="s")
        )
        self.trades = self.parse_trades(trades)
        self.compute_results()

    def compute_results(self) -> None:
        self.results["Starting Equity [$]"] = self.equity_curve.values[0]
        self.results["Ending Equity [$]"] = self.equity_curve.values[-1]
        self.results["Trades"] = self.trades
        self.results["Equity Curve [$]"] = self.equity_curve
        self.results["Trading Duration [d]"] = self.total_trade_duration()
        self.results["Total Returns [%]"] = (
            (self.results["Ending Equity [$]"] / self.results["Starting Equity [$]"])
            - 1
        ) * 100
        self.results["CAGR (Ann.) [%]"] = self.cagr(
            self.results["Starting Equity [$]"],
            self.results["Ending Equity [$]"],
            self.total_trade_duration(result_unit="years"),
        )
        self.results["Volatility (Ann.) [%]"] = self.volatility(self.equity_curve)
        self.results["Portfolio Sharpe Ratio"] = self.sharpe(
            self.results["CAGR (Ann.) [%]"], self.results["Volatility (Ann.) [%]"]
        )
        self.results["Max Drawdown [%]"] = self.max_drawdown(self.equity_curve)
        self.results["Number of Trades"] = self.trades.shape[0]
        self.results.update(self.win_stats(self.results["Trades"]))
        self.results["Expected Value [%]"] = self.expected_value(
            self.results["Win Rate [%]"],
            self.results["Avg Win [%]"],
            self.results["Avg Loss [%]"],
        )
        self.results.update(self.trade_duration_stats(self.results["Trades"]))

        # Compute all results above
        kpis = self.results
        trades: pd.DataFrame = kpis.pop("Trades")
        eq_curve: pd.DataFrame = pd.DataFrame(
            kpis.pop("Equity Curve [$]"), columns=["Equity [$]"]
        )
        kpis: pd.DataFrame = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
        kpis = kpis.set_index("KPI")
        self.results = {"KPIs": kpis, "Trades": trades, "Equity Curve [$]": eq_curve}

    def total_trade_duration(self, result_unit: str = "days") -> float:
        """result_unit: str can be any of the following: ["minutes", "hours", "days", "years"]"""
        if self.equity_curve.shape[0] == 0:
            return 0
        start: datetime = self.equity_curve.index[0]
        end: datetime = self.equity_curve.index[-1]
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

    @staticmethod
    def parse_trades(trades: list[Trade]) -> pd.DataFrame:
        trades_list: list[list] = list(
            map(
                lambda trade: [
                    trade.symbol,
                    trade.order_type,
                    trade.size,
                    trade.opening_price,
                    trade.closing_price,
                    pd.to_datetime(trade.opening_datetime / 1e9, unit="s"),
                    pd.to_datetime(trade.closing_datetime / 1e9, unit="s"),
                    trade.commission,
                ],
                trades,
            )
        )
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
            df["Order Type"].isin({"SELL", "SELL_LIMIT"}),
            "Directional Size",
        ] *= -1
        df["Gross PnL"] = df["Directional Size"] * (
            df["Closing Price"] - df["Opening Price"]
        )
        df["Net PnL"] = df["Gross PnL"] - df["Commission"]
        df["Net PnL [%]"] = (df["Net PnL"] / (df["Opening Price"] * df["Size"])) * 100
        return df

    @staticmethod
    def cagr(starting_cash: float, ending_cash: float, trade_dur_years: float) -> float:
        if trade_dur_years == 0:
            return np.nan
        return (((ending_cash / starting_cash) ** (1 / trade_dur_years)) - 1) * 100

    @staticmethod
    def volatility(equity_curve: pd.Series) -> float:
        return equity_curve.pct_change().std() * np.sqrt(252) * 100

    @staticmethod
    def sharpe(cagr: float, volatility: float) -> float:
        if volatility == 0:
            return np.nan
        return cagr / volatility

    @staticmethod
    def win_stats(trades: pd.DataFrame) -> dict[str, float]:
        df = trades.copy(deep=True)
        res: dict[str, float] = {}
        res["Win Rate [%]"] = ((df["Net PnL"] > 0).sum() / df.shape[0]) * 100

        res["Avg Win [%]"] = df.loc[(df["Net PnL [%]"] > 0), "Net PnL [%]"].mean()

        res["Avg Loss [%]"] = df.loc[(df["Net PnL [%]"] <= 0), "Net PnL [%]"].mean()

        res["Best Win [%]"] = df.loc[(df["Net PnL [%]"] > 0), "Net PnL [%]"].max()

        res["Worst Loss [%]"] = df.loc[(df["Net PnL [%]"] <= 0), "Net PnL [%]"].min()

        res["Profit Factor"] = np.nan

        if df.loc[(df["Net PnL"] <= 0), "Net PnL [%]"].sum() != 0:
            res["Profit Factor"] = df.loc[
                (df["Net PnL"] > 0), "Net PnL"
            ].sum() / np.abs(df.loc[(df["Net PnL"] <= 0), "Net PnL"].sum())

        return res

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        eq_curve: pd.Series = equity_curve.copy(deep=True)
        returns: pd.Series = eq_curve.pct_change()
        cum_returns: pd.Series = (1 + returns).cumprod()
        cum_roll_max: pd.Series = cum_returns.cummax()
        drawdown: pd.Series = cum_roll_max - cum_returns
        return (drawdown / cum_roll_max).max() * 100

    @staticmethod
    def expected_value(
        win_rate_perc: float, avg_win_perc: float, avg_loss_perc: float
    ) -> float:
        return (
            (win_rate_perc * avg_win_perc) + ((100 - win_rate_perc) * avg_loss_perc)
        ) / 100

    @staticmethod
    def trade_duration_stats(trades: pd.DataFrame) -> dict[str, pd.Timestamp]:
        df: pd.DataFrame = trades.copy(deep=True)
        return {
            "Avg Trade Duration [d]": (
                df["Closing Datetime"] - df["Opening Datetime"]
            ).mean(),
            "Max Trade Duration [d]": (
                df["Closing Datetime"] - df["Opening Datetime"]
            ).max(),
        }
