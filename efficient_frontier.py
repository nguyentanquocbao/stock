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

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed


def efficient_frontier(
    returns: pd.DataFrame,
    market_df: pd.DataFrame,
    num_portfolios: int = 100,
    risk_free_rate: float = 0.01,
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

    def portfolio_statistics(weights, returns):
        """Calculate portfolio statistics (return, volatility, Sharpe ratio)"""
        portfolio_return = (returns.multiply(weights, axis=1).sum(axis=1)).iloc[
            -1
        ]  # Using iloc instead of [-1]
        portfolio_std = returns.multiply(weights, axis=1).sum(axis=1).std()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    def minimize_volatility(weights):
        """Objective function to minimize volatility"""
        return portfolio_statistics(weights, returns)[1]

    # market computation
    total_market_value = market_df.groupby("time")["market_value"].transform("sum")
    market_df["market_scale"] = market_df["market_value"] / total_market_value
    market_df["market_log_return"] = market_df["market_scale"] * (market_df["log_return"] + 1)
    market_df = market_df.pivot_table(index="time", values="market_log_return", columns="ticker")
    market_df = market_df.sum(axis=1).to_frame(name="market_log_return")
    market_df = market_df.cumprod()
    market_std = market_df.std()
    market_return = market_df.iloc[-1]

    # Porfolio computation
    returns = (1 + returns).cumprod()
    # Constraints
    num_assets = len(returns.columns)
    bounds = tuple(range0 for _ in range(num_assets))  # weights between given range

    # Initial guess (equal weights)
    init_weights = np.array([1 / num_assets] * num_assets)

    # Find portfolio with maximum Sharpe ratio
    # optimal_sharpe = minimize(
    #     minimize_negative_sharpe,
    #     init_weights,
    #     method="SLSQP",
    #     bounds=bounds,
    #     constraints=constraints,
    # )

    # Calculate efficient frontier
    target_returns = np.linspace(returns.min().min(), returns.max().max(), num_portfolios)

    efficient_portfolios = []

    def compute_efficient_portfolio(target):
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
            return_val, std_val, sharpe_val = portfolio_statistics(result.x, returns)
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

    # Find min Sharpe ratio portfolio
    min_sharpe_idx = np.argmin(results_array[:, 2])
    min_sharpe_portfolio = results_array[min_sharpe_idx]

    # Create the scatter plot (existing code)
    plt.scatter(
        results_array[:, 1],
        results_array[:, 0],
        c=results_array[:, 2],
        cmap="viridis",
        marker="o",
        s=10,
        alpha=0.3,
    )

    # Add max Sharpe point
    plt.scatter(
        max_sharpe_portfolio[1],
        max_sharpe_portfolio[0],
        color="red",
        marker="*",
        s=200,
        label="Maximum Sharpe ratio",
    )

    # Add min Sharpe point
    plt.scatter(
        min_sharpe_portfolio[1],
        min_sharpe_portfolio[0],
        color="yellow",
        marker="*",
        s=200,
        label="Minimum Sharpe ratio",
    )
    plt.scatter(
        market_std,
        market_return,
        c=market_return / market_std,
        cmap="viridis",
        marker="o",
        s=200,
        label="market",
    )

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
        },
        "Minimum Sharpe": {
            "Return": min_sharpe_portfolio[0],
            "Volatility": min_sharpe_portfolio[1],
            "Weights": dict(zip(returns.columns, min_sharpe_portfolio[3:])),
        },
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
