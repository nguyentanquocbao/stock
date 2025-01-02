"""
efficient_frontier.py

This module provides functionality to plot the efficient frontier 
    for a given set of asset returns and market values.
The efficient frontier is a concept in modern portfolio theory that 
    represents the set of portfolios that maximize
expected return for a given level of risk.

Functions:
- plot_efficient_frontier(returns_df, market_value_df, n_portfolios=1000,
    risk_free_rate=0.02): Plots the efficient frontier
    including the market portfolio point.

Dependencies:
- matplotlib.pyplot
- numpy
- pandas
"""

import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize


def summary_market(df):
    """
    Calculate yearly market statistics based on market value and log returns.

    This function processes financial data to compute market-weighted log returns and their statistics
    on a yearly basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the following columns:
        - time: datetime column for the time period
        - market_value: numeric column representing market value
        - log_return: numeric column representing logarithmic returns
        - ticker: string column representing stock identifiers

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing yearly market statistics with columns:
        - year: int, the year of the statistics
        - market_log_return: float, the cumulative market-weighted log return for the year
        - std: float, the standard deviation of market-weighted log returns for the year
    """
    total_market_value = df.groupby("time")["market_value"].transform("sum")
    df["market_scale"] = df["market_value"] / total_market_value
    df["market_log_return"] = df["market_scale"] * (df["log_return"] + 1)
    df = df.pivot_table(index="time", values="market_log_return", columns="ticker")
    df["year"] = df.reset_index()["time"].dt.year
    df = df.sum(axis=1).to_frame(name="market_log_return").reset_index()
    df["year"] = df["time"].dt.year
    df["market_log_return"] = df.groupby("year")["market_log_return"].cumprod()
    df_stat = df.groupby("year").agg({"market_log_return": ["last", "std"]}).reset_index()
    df_stat.columns = ["year", "market_log_return", "std"]
    return df_stat, df


def calculate_efficient_frontier_points(points_array, risk_free_rate):
    """
    Calculate the efficient frontier points using the maximum slope algorithm.

    Args:
        points_array: numpy array with shape (n, 2) containing (volatility, return) pairs
        risk_free_rate: float representing the risk-free rate

    Returns:
        tuple: (frontier_x, frontier_y) lists containing the frontier points
    """
    frontier_x = [0]
    frontier_y = [1 + risk_free_rate]

    while True:
        max_slope = float("-inf")
        max_point = None

        # Find point that creates highest slope from last frontier point
        for point in points_array:
            if point[0] == frontier_x[-1] and point[1] == frontier_y[-1]:
                continue
            # Calculate slope
            slope = (point[1] - frontier_y[-1]) / (point[0] - frontier_x[-1])
            if slope > max_slope:
                max_slope = slope
                max_point = point

        # If no new point found or we've reached the end, break
        if max_point is None or max_point[0] <= frontier_x[-1]:
            break

        # Add new point to frontier
        frontier_x.append(max_point[0])
        frontier_y.append(max_point[1])

    return frontier_x, frontier_y


def portfolio_statistics(weights, returns, risk_free_rate=0.05):
    """Calculate portfolio statistics (return, volatility, Sharpe ratio)"""
    portfolio_return = (returns.multiply(weights, axis=1).sum(axis=1)).iloc[
        -1
    ]  # Using iloc instead of [-1]
    returns = returns.multiply(weights, axis=1).sum(axis=1)
    portfolio_std = returns.std()
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio, returns


def efficient_frontier(
    returns: pd.DataFrame,
    market_df: pd.DataFrame,
    num_portfolios: int = 100,
    risk_free_rate: float = 0.05,
    range0=(0.01, 1),
):
    """
    Calculate the efficient frontier using scipy.minimize

    Parameters:
    returns (DataFrame): DataFrame of asset returns
    num_portfolios (int): Number of portfolios to generate
    risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation
    range0 (tuple): Range for portfolio weights, default (0.01, 1) to avoid division by zero

    Returns:
    tuple: (results_array, optimal_sharpe_portfolio)
    """

    def minimize_volatility(weights):
        """Objective function to minimize volatility"""
        return portfolio_statistics(weights, returns)[1]

    market_stat, _ = summary_market(market_df)
    market_std = market_stat.iloc[-1]["std"]
    market_return = market_stat.iloc[-1]["market_log_return"]
    # Porfolio computation
    returns = (1 + returns).cumprod()
    # Constraints
    num_assets = len(returns.columns)
    bounds = tuple(range0 for _ in range(num_assets))  # weights between given range

    # Initial guess (equal weights)
    init_weights = np.array([1 / num_assets] * num_assets)
    target_returns = np.linspace(returns.min().min(), returns.max().max(), num_portfolios)
    efficient_portfolios = []

    def compute_efficient_portfolio(target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            constraints = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {
                    "type": "eq",
                    "fun": lambda x, target=target: portfolio_statistics(x, returns)[0] - target,
                },
            )
            result = minimize(
                minimize_volatility,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )

            if result.success:
                return_val, std_val, sharpe_val, _ = portfolio_statistics(
                    result.x, returns, risk_free_rate
                )
                return [return_val, std_val, sharpe_val] + list(result.x)
            else:
                return None

    efficient_portfolios = Parallel(n_jobs=-7)(
        delayed(compute_efficient_portfolio)(target) for target in target_returns
    )

    # Filter out None results
    efficient_portfolios = [
        portfolio for portfolio in efficient_portfolios if portfolio is not None
    ]

    # Check if we have any valid portfolios

    # Convert results to array
    results_array = np.array(efficient_portfolios)

    # Debug print
    print(f"Results array shape: {results_array.shape}")

    if len(results_array.shape) < 2:
        print("Invalid results array shape!")
        return None, None

    plt.figure(figsize=(10, 6))

    max_sharpe_idx = np.argmax(results_array[:, 2])  # column 2 contains Sharpe ratios
    max_sharpe_portfolio = results_array[max_sharpe_idx]

    # Create the scatter plot (existing code)
    plt.scatter(
        results_array[:, 1],
        results_array[:, 0],
        c=results_array[:, 2],
        cmap="viridis",
        marker="o",
        s=4,
        alpha=0.3,
    )

    plt.scatter(
        max_sharpe_portfolio[1],
        max_sharpe_portfolio[0],
        color="green",
        marker="p",
        s=100,
        label="Maximum Sharpe ratio",
    )

    # Add dashed lines to axes
    plt.axhline(y=max_sharpe_portfolio[0], color="green", linestyle="--", alpha=0.3)
    plt.axvline(x=max_sharpe_portfolio[1], color="green", linestyle="--", alpha=0.3)

    plt.scatter(market_std, market_return, color="red", marker="p", s=100, label="market")
    # Add dashed lines to axes for market point
    plt.axhline(y=market_return, color="red", linestyle="--", alpha=0.3)
    plt.axvline(x=market_std, color="red", linestyle="--", alpha=0.3)

    # Convert results to points and calculate frontier
    points = np.column_stack((results_array[:, 1], results_array[:, 0]))
    frontier_x, frontier_y = calculate_efficient_frontier_points(points, risk_free_rate)

    # Plot the efficient frontier

    # Extend the frontier line to the right
    extended_x = frontier_x + [max(frontier_x[-1] * 1.5, results_array[:, 1].max() * 1.2)]
    extended_y = frontier_y + [
        frontier_y[-1]
        + (frontier_y[-1] - frontier_y[-2])
        * (extended_x[-1] - frontier_x[-1])
        / (frontier_x[-1] - frontier_x[-2])
    ]
    plt.plot(
        extended_x, extended_y, "r-", linewidth=2, alpha=0.8, label="CAL - Capital Allocation Line"
    )
    plt.scatter(
        frontier_x[-1], frontier_y[-1], color="blue", marker="p", s=100, label="CAL tangent"
    )
    # Plot horizontal and vertical lines at the tangent point
    plt.axhline(y=frontier_y[-1], color="blue", linestyle="--", alpha=0.3)
    plt.axvline(x=frontier_x[-1], color="blue", linestyle="--", alpha=0.3)

    plt.colorbar(label="Sharpe ratio")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.legend()

    # Add portfolio composition text
    portfolio_metrics = {
        "Maximum Sharpe": {
            "Return": max_sharpe_portfolio[0],
            "Volatility": max_sharpe_portfolio[1],
            "Weights": dict(zip(returns.columns, max_sharpe_portfolio[3:])),
        }
    }

    return portfolio_metrics, [market_return, market_std]


def plot_portfolio_weights(weights: Dict, title="Portfolio Weights"):
    """
    Plots the portfolio weights on a radar chart.
    Parameters:
    weights (pd.Series): A pandas Series containing the portfolio weights
        with asset names as the index.
    title (str, optional): The title of the plot. Defaults to "Portfolio Weights".
    Returns:
    None: This function does not return any value.
        It displays a radar chart of the portfolio weights.
    """
    # Define the labels and number of variables

    labels = list(weights.keys())
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = list(np.linspace(0, 2 * np.pi, num_vars, endpoint=False))

    # Close the plot by appending first values
    values = list(weights.values())
    values += values[:1]
    angles += angles[:1]

    # Create the radar chart
    _, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="blue", alpha=0.25)
    ax.plot(angles, values, color="blue", linewidth=2)

    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title(title)
    plt.show()
