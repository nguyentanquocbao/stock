"""Market Visualization Module

This module provides functionality for visualizing and analyzing market data across different
exchanges. It includes tools for calculating and plotting various market statistics such as
Sharpe ratios, returns, and cumulative returns.

The main component is the MarketVisualize class which processes market data and creates
visualizations to help understand market performance over time.

Key Features:
    - Calculate and visualize Sharpe ratios (both regular and logarithmic returns)
    - Analyze market returns and cumulative log returns across exchanges
    - Generate yearly market statistics summaries
    - Compare performance across different exchanges

Example Usage:
    >>> import pandas as pd
    >>> from visualize import MarketVisualize
    >>> market_data = pd.read_csv('market_data.csv')
    >>> viz = MarketVisualize(market_data, risk_free_rate=0.02)
    >>> viz.plot_sharp_ratios()
    >>> viz.plot_returns()

Required Dependencies:
    - pandas
    - numpy
    - matplotlib

Notes:
    The market data should contain columns for time, market return, market log return,
    and exchange. The time series data should be in datetime format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MarketVisualize:
    """A class for visualizing market data and calculating market statistics.
    This class provides functionality to process and visualize market data, including
    calculating various market statistics such as Sharpe ratios, excess returns, and
    cumulative returns across different exchanges over time.
    Parameters
    ----------
    market_data : pd.DataFrame
        A DataFrame containing market data with the following required columns:
        - time: datetime column for the time period
        - return_weighted: column containing market returns
        - log_return_weighted: column containing logarithmic market returns
        - exchange: column identifying different exchanges
    risk_free_rate : float, optional
        The risk-free rate used in calculations (default is 0.02 or 2%)
    Attributes
    ----------
    market_data : pd.DataFrame
        The input market data DataFrame
    risk_free_rate : float
        The risk-free rate used in calculations
    summary : pd.DataFrame
        Processed DataFrame containing summarized market statistics by year and exchange,
        including:
        - market excess return standard deviation
        - average market excess return
        - cumulative market excess log return
        - Sharpe ratios for both regular and log returns
    Methods
    -------
    _market_clean()
        Internal method to process market data and calculate summary statistics
    plot_sharp_ratios()
        Visualizes Sharpe ratios over time for different exchanges
    Examples
    --------
    >>> market_viz = MarketVisualize(market_data, risk_free_rate=0.02)
    >>> market_viz.plot_sharp_ratios()
    """

    def __init__(self, market_data: pd.DataFrame, risk_free_rate=0.02):
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate
        self.summary = self._market_clean()

    def _market_clean(self) -> pd.DataFrame:
        """
        Summarize market statistics each year.
        This function calculates the standard deviation of market returns, the average market return,
        and the cumulative market log return at the end of each year.
        Returns:
            pd.DataFrame: DataFrame with summarized market statistics for each year.
        """
        # Filter out current year data
        current_year = pd.Timestamp.now().year
        self.market_data = self.market_data.loc[
            self.market_data["time"].dt.year < current_year
        ].copy()
        self.market_data = self.market_data.loc[self.market_data["exchange"] != "DELISTED"]
        self.market_data["year"] = pd.to_datetime(self.market_data["time"]).dt.year
        self.market_data["cumulative_log_return"] = (
            (self.market_data["log_return_weighted"] + 1)
            .groupby([self.market_data["exchange"], self.market_data["year"]])
            .cumprod()
        )
        summary = (
            self.market_data.groupby(["exchange", "year"])
            .agg(
                return_std=("return_weighted", "std"),
                return_weighted=("return_weighted", "mean"),
                cumu_log_return=("log_return_weighted", "sum"),
                cumu_log_return_std=("cumulative_log_return", "std"),
            )
            .reset_index()
        )
        summary["return_weighted"] = summary["return_weighted"] * 252 - self.risk_free_rate
        summary["cumu_log_return"] = summary["cumu_log_return"] - self.risk_free_rate
        summary["sharp_ratio_return"] = summary["return_weighted"] / summary["return_std"]
        summary["sharp_ratio_log_return"] = (
            summary["cumu_log_return"] / summary["cumu_log_return_std"]
        )
        return summary

    def plot_sharp_ratios(self):
        """
        Plot sharp ratios for each exchange over the years.
        """
        summary = self.summary
        _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        exchanges = summary["exchange"].unique()
        for exchange in exchanges:
            exchange_data = summary[summary["exchange"] == exchange]
            ax[0].plot(exchange_data["year"], exchange_data["sharp_ratio_return"], label=exchange)
            ax[1].plot(
                exchange_data["year"], exchange_data["sharp_ratio_log_return"], label=exchange
            )

        ax[0].set_title("Sharp Ratio Return Over Years")
        ax[0].set_ylabel("Sharp Ratio Return")
        ax[0].legend()

        ax[1].set_title("Sharp Ratio Log Return Over Years")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("Sharp Ratio Log Return")
        ax[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Add padding at the bottom
        plt.figtext(
            0.02,
            0.05,
            "Note: Return Sharpe ratio uses annualized daily returns (x252).\n"
            "Log return Sharpe ratio uses cumulative daily returns.",
            style="italic",
            bbox={"facecolor": "yellow", "alpha": 0.2},
        )
        plt.show()
        return summary

    def plot_returns(self) -> pd.DataFrame:
        """
        Plot yearly returns and cumulative log returns for each exchange,
        with standard deviations on secondary axes.
        """
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        exchanges = self.summary["exchange"].unique()

        # Create twin axes for standard deviations
        ax1_twin = ax1.twinx()
        ax2_twin = ax2.twinx()

        # Use a color cycle so each exchange has a unique color
        color_cycle = plt.rcParams["axes.prop_cycle"]()

        for exchange in exchanges:
            color = next(color_cycle)["color"]
            exchange_data = self.summary[self.summary["exchange"] == exchange]

            ax1.plot(
                exchange_data["year"],
                exchange_data["return_weighted"],
                label=f"{exchange} return",
                color=color,
            )
            ax1_twin.plot(
                exchange_data["year"],
                exchange_data["return_std"],
                label=f"{exchange} std",
                color=color,
                linestyle="--",
            )
            ax2.plot(
                exchange_data["year"],
                exchange_data["cumu_log_return"],
                label=f"{exchange} return",
                color=color,
            )
            ax2_twin.plot(
                exchange_data["year"],
                exchange_data["cumu_log_return_std"],
                label=f"{exchange} std",
                color=color,
                linestyle="--",
            )

        ax1.set_title("Annualized Market Returns Over Years")
        ax1.set_ylabel("Market Return")
        ax1_twin.set_ylabel("Standard Deviation")

        ax2.set_title("Cumulative Log Returns Over Years")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cumulative Log Return")
        ax2_twin.set_ylabel("Standard Deviation")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc="upper left")

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc="upper left")

        plt.tight_layout()
        plt.show()
        return self.summary
