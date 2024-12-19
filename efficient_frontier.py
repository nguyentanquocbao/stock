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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def efficient_frontier(
    returns: pd.DataFrame, num_portfolios: int = 100, risk_free_rate: float = 0.01, range0=(0, 1)
):
    """
    Calculate the efficient frontier using scipy.minimize

    Parameters:
    returns (DataFrame): DataFrame of asset returns
    num_portfolios (int): Number of portfolios to generate
    risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation

    Returns:
    tuple: (results_array, optimal_sharpe_portfolio)
    """

    def portfolio_statistics(weights, returns):
        """Calculate portfolio statistics (return, volatility, Sharpe ratio)"""
        portfolio_return = returns.multiply(weights, axis=1).sum(axis=1)[-1]
        portfolio_std = returns.multiply(weights, axis=1).sum(axis=1).std()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    def minimize_volatility(weights):
        """Objective function to minimize volatility"""
        return portfolio_statistics(weights, returns)[1]

    def minimize_negative_sharpe(weights):
        """Objective function to maximize Sharpe ratio"""
        return -portfolio_statistics(weights, returns)[2]

    returns = (1 + returns).cumprod()
    # Constraints
    num_assets = len(returns.columns)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # weights sum to 1
    bounds = tuple(range0 for _ in range(num_assets))  # weights between 0 and 1

    # Initial guess (equal weights)
    init_weights = np.array([1 / num_assets] * num_assets)

    # Find portfolio with maximum Sharpe ratio
    optimal_sharpe = minimize(
        minimize_negative_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # Calculate efficient frontier
    target_returns = np.linspace(returns.mean().min(), returns.mean().max(), num_portfolios)

    efficient_portfolios = []

    for target in target_returns:
        # Additional constraint for target return
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
        )

        if result.success:
            return_val, std_val, sharpe_val = portfolio_statistics(result.x, returns)
            efficient_portfolios.append([return_val, std_val, sharpe_val] + list(result.x))

    # Convert results to array
    results_array = np.array(efficient_portfolios)
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

    return portfolio_metrics
