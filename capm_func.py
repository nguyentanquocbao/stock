"""
Capital Asset Pricing Model (CAPM) Analysis Module

This module provides functions for calculating and visualizing CAPM metrics for multiple stocks.
It includes parallel processing capabilities and supports rolling window analysis.

Functions:
    plot_capm_results(test):
        Plots CAPM results including beta coefficients, alpha values, and estimated returns 
        for multiple tickers over time.
        
    capm(market_data, stock_data, window=1, risk_free_rate=0.05, para=True):
        Calculates CAPM metrics including beta, market risk premium, estimated returns 
        and alpha for given stock data.

Dependencies:
    - numpy
    - pandas 
    - joblib (for parallel processing)
    - matplotlib

Notes:
    - The module uses rolling window approach for time-varying beta calculations
    - Supports both parallel and sequential processing modes
    - Includes visualization capabilities for CAPM metrics over time
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def plot_capm_results(test):
    """
        Plots CAPM (Capital Asset Pricing Model) results for multiple tickers.
        This function creates a multi-panel figure where each panel corresponds to a unique ticker
        and displays the evolution of beta coefficients, alpha values, and estimated returns over time.
        Parameters
        ----------
        test : pandas.DataFrame
            A DataFrame containing the following columns:
            - time : datetime
                Timestamps for the data points
            - ticker : str
                Stock ticker symbols
            - beta : float
                Beta coefficients from CAPM regression
            - alpha : float
                Alpha values from CAPM regression
            - estimated_return : float
                Estimated returns from CAPM model
        Returns
        -------
        None
            Displays the plot using matplotlib's plt.show()
        Notes
    -----
        - Each subplot contains:
            * Beta values plotted as a blue line (left y-axis)
            * Alpha values plotted as red bars (right y-axis)
            * Estimated returns plotted as green bars (right y-axis)
            * Horizontal reference lines at y=0 for both axes
        - The figure size is set to (15, 150) and includes tight layout padding
        - Years are extracted from timestamps for x-axis grouping
    """
    # Convert datetime to string format for better bar grouping
    test["year"] = test["time"].dt.year
    fig, axes = plt.subplots(len(test["ticker"].unique()), 1, figsize=(15, 150))
    fig.tight_layout(pad=5.0)

    # Plot for each ticker
    for i, ticker in enumerate(test["ticker"].unique()):
        ticker_data = test[test["ticker"] == ticker]

        # Create twin axes
        ax1 = axes[i]
        ax2 = ax1.twinx()

        # Plot beta on left axis
        ax1.plot(ticker_data["year"], ticker_data["beta"], color="blue", label="Beta")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Beta")

        # Plot clustered bar chart on right axis
        width = 0.4
        ax2.bar(
            ticker_data["year"] - width / 2,
            ticker_data["alpha"],
            width=width,
            color="red",
            alpha=0.5,
            label="Alpha",
        )
        ax2.bar(
            ticker_data["year"] + width / 2,
            ticker_data["estimated_return"],
            width=width,
            color="green",
            alpha=0.5,
            label="Estimated Return",
        )
        ax2.set_ylabel("Log Return")
        # Add horizontal dashed lines
        ax1.axhline(0, color="green", linestyle="--", linewidth=1, label="Zero beta Line")
        ax2.axhline(0, color="red", linestyle="--", linewidth=1, label="Zero return Line")
        # Add title and legend
        ax1.set_title(f"Ticker: {ticker}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
        )
    plt.show()


def capm(
    market_data: pd.DataFrame, stock_data: pd.DataFrame, window=1, risk_free_rate=0.05, para=True
):
    """
    Calculate Capital Asset Pricing Model (CAPM) metrics for given stock data.
    This function implements CAPM analysis by calculating beta, market risk premium,
    estimated returns and alpha for multiple stocks over specified time windows.
    It can process calculations either in parallel or sequentially.
    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame containing market returns data with columns:
        - time: datetime index
        - market_log_return: log returns of market index
    stock_data : pd.DataFrame
        DataFrame containing individual stock data with columns:
        - time: datetime index
        - ticker: stock identifier
        - log_return: log returns of stocks
    window : int, default=1
        Number of years to look back for calculations
    risk_free_rate : float, default=0.05
        Risk-free rate used in CAPM calculations
    para : bool, default=True
        If True, performs calculations in parallel using joblib
        If False, performs calculations sequentially
    Returns
    -------
    pd.DataFrame
        DataFrame containing CAPM metrics with columns:
        - time: datetime of calculation
        - ticker: stock identifier
        - beta: calculated beta coefficient
        - market_risk_premium: market return minus risk-free rate
        - log_return: actual stock return
        - past_beta: previous period's beta (when para=True)
        - estimated_return: CAPM estimated return (when para=True)
        - alpha: excess return over CAPM estimate (when para=True)
    Notes
    -----
    The function uses rolling window approach to calculate time-varying betas
    and other CAPM metrics on a year-by-year basis.
    """

    def calculate_capm_metrics(ticker_data, market_returns, end_date, start_date):
        # Get data up to end date
        ticker_data = ticker_data[
            (ticker_data["time"] >= start_date) & (ticker_data["time"] <= end_date)
        ]
        ticker_data.loc[:, "log_return"] = (ticker_data["log_return"] + 1).cumprod()
        ticker_data1 = ticker_data.merge(market_returns, how="inner", on="time")
        if ticker_data1.shape[0] != ticker_data.shape[0]:
            print("lack of data")
        window_returns = ticker_data1["log_return"]
        window_market = ticker_data1["market_log_return"]

        beta = np.cov(window_returns, window_market)[0, 1] / np.var(window_market)

        # Calculate CAPM expected return
        market_risk_premium = window_market.iloc[-1] - risk_free_rate
        # estimated_return = risk_free_rate + beta * market_risk_premium

        # Calculate alpha
        actual_return = window_returns.iloc[-1]
        # alpha = actual_return - estimated_return

        return {
            "time": end_date,
            "ticker": ticker_data["ticker"].iloc[0],
            "beta": beta,
            "market_risk_premium": market_risk_premium,
            # "alpha": alpha,
            "log_return": actual_return,
        }

    # Process each ticker and year end date in parallel
    tt = stock_data.groupby([stock_data["ticker"], stock_data["time"].dt.year]).agg({"time": "max"})
    tt.columns = ["max_time"]
    tt["min_time"] = tt["max_time"] - pd.offsets.YearBegin(window)  # type: ignore
    tt.reset_index(inplace=True)
    if para:
        results = Parallel(n_jobs=-1)(
            delayed(calculate_capm_metrics)(
                stock_data.loc[stock_data["ticker"] == smalljob[0]].dropna(subset=["log_return"]),
                market_data[["time", "market_log_return"]],
                smalljob[1],
                smalljob[2],
            )
            for smalljob in zip(tt["ticker"], tt["max_time"], tt["min_time"])
        )
        # Combine results
        results = pd.DataFrame(list(results))
        results["past_beta"] = results.groupby("ticker")["beta"].shift(1)
        results["estimated_return"] = results["market_risk_premium"] * results["past_beta"] + 0.05
        results["alpha"] = results["log_return"] - results["estimated_return"]
        return results
    else:
        out = pd.DataFrame()
        for smalljob in zip(tt["ticker"], tt["max_time"], tt["min_time"]):
            # print(smalljob)
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            calculate_capm_metrics(
                                stock_data[stock_data["ticker"] == smalljob[0]].dropna(
                                    subset=["log_return"]
                                ),
                                market_data[["time", "market_log_return"]],
                                smalljob[1],
                                smalljob[2],
                            )
                        ]
                    ),
                ],
                axis=0,
            )
        return out
