"""
Market Visualization Module

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
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import logging
from stock_data import StockData
from dataclasses import dataclass
import numpy as np
from matplotlib.collections import LineCollection

@dataclass
class MarketVisualize(StockData):
    """A class for visualizing market data and calculating market statistics.
    This class provides functionality to process and visualize market data, including
    calculating various market statistics such as Sharpe ratios, excess returns, and
    cumulative returns across different exchanges over time.
    Parameters
    ----------
    market_data : pd.DataFrame
        A DataFrame containing market data with the following required columns:
        - time: datetime column for the time period
        - return: column containing market returns
        - log_return: column containing logarithmic market returns
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
    risk_free_rate: float = 0.03
    remove_current_year: bool = True
    def __post_init__(self):
        super().__post_init__()
        self.stocks_data, self.market_data = self.get_data()
        self._market_clean()
        self.summary()



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
        if self.remove_current_year:
            self.market_data = self.market_data.loc[
                self.market_data["time"].dt.year < current_year
            ].copy()
        self.market_data = self.market_data.loc[self.market_data["ticker"] != "DELISTED"]
        
    def summary(self):
        summary=self.market_data.copy()
        summary['cumu_log_return']=summary.groupby(["ticker",'year'])["log_return"].cumsum()

        summary = (
            summary.groupby(["ticker", "year"])
            .agg(
                return_std=("return", "std"),
                return_mean=("return", "mean"),
                cumu_log_return=("log_return", "sum"),
                cumu_log_return_std=("cumu_log_return", "std"),
            )
            .reset_index()
        )
        summary['return_mean']=summary["return_mean"] * 252
        summary["exceed_return"] = summary["return_mean"] - self.risk_free_rate
        summary["cumu_exceed_return"] = summary["cumu_log_return"] - self.risk_free_rate
        summary["sharp_ratio_return"] = summary["exceed_return"] / summary["return_std"]
        summary["sharp_ratio_log_return"] = (
            summary["cumu_exceed_return"] / summary["cumu_log_return_std"]
        )
        self.summary=summary

    

    def plot_sharp_ratios(self,min_year=2013):
        """
        Plot sharp ratios for each exchange over the years.
        """
        summary=self.summary
        if min_year:        
            summary=summary[summary["year"]>min_year]
        
        _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        sns.lineplot(data=summary, x="year", y="sharp_ratio_return", hue="ticker", ax=ax[0])
        sns.lineplot(data=summary, x="year", y="sharp_ratio_log_return", hue="ticker", ax=ax[1])
        plt.show()


    def plot_risk_and_returns(self,min_year=2013) -> pd.DataFrame:
        """
        Plot yearly returns and cumulative log returns for each exchange,
        with standard deviations on secondary axes.
        """
        summary=self.summary.copy()
        if min_year:        
            summary=summary[summary["year"]>min_year]
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot for return
        sns.lineplot(
            data=summary,
            x="year",
            y="return_mean",
            hue="ticker",
            ax=ax1,
            color="blue",
            alpha=0.6,
        )

        ax2 = ax1.twinx()
        sns.lineplot(
            data=summary,
            x="year",
            y="return_std",
            hue="ticker",
            ax=ax2,
            color="red",
            marker="o",
            linestyle="--",
        )
        ax2.set_ylabel("Return Std")
        ax1.set_title("Return Weighted and Std by Year")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Return Weighted")

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot for return
        sns.lineplot(
            data=summary,
            x="year",
            y="cumu_log_return",
            hue="ticker",
            ax=ax1,
            color="blue",
            alpha=0.6,
        )

        ax2 = ax1.twinx()
        sns.lineplot(
            data=summary,
            x="year",
            y="cumu_log_return_std",
            hue="ticker",
            ax=ax2,
            color="red",
            marker="o",
            linestyle="--",
        )
        ax2.set_ylabel("Return Std")
        ax2.set_title("Cumulative log Return Weighted and Std by Year")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Return Weighted")
    def plot_spread(self,min_year=2013):
        spread=self.market_data.copy()
        spread["spread"] = spread["high"] - spread["low"]
        spread["spread_pct"] = spread["spread"] / spread["low"]
        if min_year:
            spread=spread[spread["year"]>min_year]
        logging.info(f"số lượng index:{spread['ticker'].value_counts()}")
        plt.figure(figsize=(14, 7))
        sns.lineplot(
        data=spread, x="time", y="spread_pct", hue="ticker", palette="tab10", linewidth=0.5
    )
    def plot_scenario(self, return_col,manual_windows, min_year=2013, jump_threshold=2.0, drift_threshold=0.0005):
        def analyze_trading_cycles(df, return_col, min_window=5, max_window=252):
            """
            Analyze optimal trading cycle windows
            """
            windows = np.arange(min_window, max_window, 5)
            results = []

            for window in windows:
                # Calculate rolling statistics
                roll_vol = df[return_col].rolling(window).std()
                roll_ir = df[return_col].rolling(window).mean() / roll_vol

                # Calculate ACF
                acf = sm.tsa.stattools.acf(df[return_col].dropna(), nlags=window)[1:]

                # Signal to noise ratio
                snr = np.abs(np.mean(acf)) / np.std(acf)

                results.append(
                    {
                        "window": window,
                        "vol_stability": roll_vol.std(),
                        "mean_ir": roll_ir.mean(),
                        "signal_noise": snr,
                        "significant_lags": sum(np.abs(acf) > 2 / np.sqrt(len(df))),
                    }
                )
            results_df = pd.DataFrame(results)
            # Score windows based on metrics
            results_df["total_score"] = (
                results_df["mean_ir"].rank()
                + results_df["signal_noise"].rank()
                + results_df["significant_lags"].rank()
                - results_df["vol_stability"].rank()
            )

            return results_df.sort_values("total_score", ascending=False)
        def detect_scenario(df, return_col,exchange,manual_windows ):
            if not manual_windows:
                window = analyze_trading_cycles(df, return_col)["window"].values[0]
            else:
                window=manual_windows
            logging.info(f"Optimal window for exchange {exchange}: {window}")
            df["rolling_mean"] = df[return_col].rolling(window).mean()
            df["rolling_std"] = df[return_col].rolling(window).std()

            # Detect jumps if absolute change > jump_threshold * rolling_std
            df["jump"] = (df[return_col] - df["rolling_mean"]).abs() > jump_threshold * df[
                "rolling_std"
            ]

            # Estimate drift by rolling linear regression slope
            def slope(x):
                y = x.values
                x_arr = np.arange(len(y))
                if len(y) < 2:
                    return 0
                slope_ = np.polyfit(x_arr, y, 1)[0]
                return slope_

            df["rolling_slope"] = df[return_col].rolling(window).apply(slope, raw=False)
            df["drift"] = df["rolling_slope"].abs() > drift_threshold

            # Classify scenario
            # 1) Jump: if "jump" is True
            # 2) Drift: if not jump but "drift" is True
            # 3) No-drift: otherwise
            conditions = [df["jump"], ~df["jump"] & df["drift"], ~df["jump"] & ~df["drift"]]
            scenarios = ["jump", "drift", "no_drift"]

            df["scenario"] = np.select(conditions, scenarios, default="no_drift")
            return df
        df = self.market_data[self.market_data["year"] > min_year]
        out = pd.DataFrame()
        exchanges = df['ticker'].unique()
        
        # Process all exchanges first
        for i in exchanges:
            df_exchange = df[df["ticker"] == i].copy()
            if len(df_exchange) > 0:
                scenario_data = detect_scenario(df_exchange, return_col,i,manual_windows)
                out = pd.concat([out, scenario_data])
        
        # Create single figure with subplots
        _, axes = plt.subplots(len(exchanges), 1, figsize=(15, 5*len(exchanges)))
        if len(exchanges) == 1:
            axes = [axes]
        
        colors = {"jump": "red", "drift": "blue", "no_drift": "green"}
        
        for idx, exchange in enumerate(exchanges):
            exchange_data = out[out["ticker"] == exchange]
            # mask = exchange_data["scenario"].isin(["no_drift", "drift", "jump"])
            points = np.array([exchange_data["time"].values, exchange_data[return_col].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            scenario_colors = exchange_data["scenario"].map(colors).iloc[1:].values
            lc = LineCollection(segments, colors=scenario_colors, linewidths=1, alpha=0.5)
            axes[idx].add_collection(lc)
            axes[idx].autoscale()
            axes[idx].set_title(f"Scenarios Over Time - {exchange}")
            axes[idx].set_ylabel("Log Return")
        
            # axes[idx].legend()
        plt.tight_layout()
        plt.show()
