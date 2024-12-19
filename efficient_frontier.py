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


def plot_efficient_frontier(
    returns_df: pd.DataFrame, market_value_df, n_portfolios=1000, risk_free_rate=0.02
):
    """
    Plot the efficient frontier including market portfolio point.

    Parameters:
    returns_df (pd.DataFrame): DataFrame of asset returns
    market_value_df (pd.DataFrame): DataFrame of market values for each asset over time
    n_portfolios (int): Number of portfolios to simulate
    risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation

    Returns:
    Plot of the efficient frontier with market portfolio
    """

    # Calculate mean returns and covariance matrix
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    # Calculate market portfolio weights based on average market values
    avg_market_values = market_value_df.mean()
    market_weights = avg_market_values / avg_market_values.sum()

    # Calculate market portfolio return and volatility
    market_return = np.sum(mean_returns * market_weights) * 252  # Annualized return
    market_std = np.sqrt(np.dot(market_weights.T, np.dot(cov_matrix * 252, market_weights)))
    market_sharpe = (market_return - risk_free_rate) / market_std

    # Lists to store returns, volatility and weights of portfolios
    returns_array = []
    volatility_array = []
    weights_array = []
    sharpe_array = []

    # Generate random portfolios
    for _ in range(n_portfolios):
        weights = np.random.random(len(returns_df.columns))
        weights = weights / np.sum(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

        returns_array.append(portfolio_return)
        volatility_array.append(portfolio_std)
        weights_array.append(weights)
        sharpe_array.append(sharpe_ratio)

    # Convert to numpy arrays
    returns_array = np.array(returns_array)
    volatility_array = np.array(volatility_array)
    sharpe_array = np.array(sharpe_array)

    # Find portfolio with highest Sharpe Ratio
    max_sharpe_idx = np.argmax(sharpe_array)
    max_sharpe_return = returns_array[max_sharpe_idx]
    max_sharpe_volatility = volatility_array[max_sharpe_idx]

    # Find portfolio with minimum volatility
    min_vol_idx = np.argmin(volatility_array)
    min_vol_return = returns_array[min_vol_idx]
    min_vol_volatility = volatility_array[min_vol_idx]

    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        volatility_array, returns_array, c=sharpe_array, cmap="viridis", marker="o", s=10, alpha=0.3
    )

    # Plot maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_volatility,
        max_sharpe_return,
        color="red",
        marker="*",
        s=200,
        label="Maximum Sharpe ratio",
    )

    # Plot minimum volatility portfolio
    plt.scatter(
        min_vol_volatility,
        min_vol_return,
        color="green",
        marker="*",
        s=200,
        label="Minimum volatility",
    )

    # Plot market portfolio
    plt.scatter(
        market_std, market_return, color="blue", marker="*", s=200, label="Market portfolio"
    )

    plt.colorbar(scatter, label="Sharpe ratio")
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with Market Portfolio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    return {
        "Maximum Sharpe Ratio Portfolio": {
            "Return": max_sharpe_return,
            "Volatility": max_sharpe_volatility,
            "Sharpe Ratio": sharpe_array[max_sharpe_idx],
            "Weights": dict(zip(returns_df.columns, weights_array[max_sharpe_idx])),
        },
        "Minimum Volatility Portfolio": {
            "Return": min_vol_return,
            "Volatility": min_vol_volatility,
            "Sharpe Ratio": sharpe_array[min_vol_idx],
            "Weights": dict(zip(returns_df.columns, weights_array[min_vol_idx])),
        },
        "Market Portfolio": {
            "Return": market_return,
            "Volatility": market_std,
            "Sharpe Ratio": market_sharpe,
            "Weights": dict(zip(returns_df.columns, market_weights)),
        },
    }


def plot_efficient_frontier_log_cum(
    returns_df, market_value_df, n_portfolios=1000, risk_free_rate=0.02
):
    """
    Plot the efficient frontier using cumulative returns.

    Parameters:
    returns_df (pd.DataFrame): DataFrame of asset returns
    market_value_df (pd.DataFrame): DataFrame of market values for each asset over time
    n_portfolios (int): Number of portfolios to simulate
    risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation

    Returns:
    Plot of the efficient frontier with market portfolio
    """

    # Calculate cumulative returns
    cum_returns = (1 + returns_df).cumprod()
    # cov_matrix = np.log(1 + returns_df).cov()
    avg_market_values = market_value_df.mean()
    market_weights = avg_market_values / avg_market_values.sum()
    market_return = cum_returns.copy()
    market_return = cum_returns.multiply(market_weights, axis=1)
    market_return = market_return[-1]
    market_std = market_return.std()
    market_return = market_return.sum()
    market_sharpe = (market_return - risk_free_rate) / market_std
    returns_array = []
    volatility_array = []
    weights_array = []
    sharpe_array = []

    # Generate random portfolios
    for _ in range(n_portfolios):
        weights = np.random.random(len(returns_df.columns))
        weights = weights / np.sum(weights)
        portfolio_return = cum_returns.multiply(weights, axis=1).mean(
            axis=1
        )  # cummulative-porfolio return each trading day
        portfolio_std = portfolio_return.std()
        portfolio_return = (
            portfolio_return.sum()
        )  # because we used log-return then we just sum all cummulative return as the last return
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        returns_array.append(portfolio_return)
        volatility_array.append(portfolio_std)
        weights_array.append(weights)
        sharpe_array.append(sharpe_ratio)

    # Convert to numpy arrays
    returns_array = np.array(returns_array)
    volatility_array = np.array(volatility_array)
    sharpe_array = np.array(sharpe_array)

    # Find portfolio with highest Sharpe Ratio
    max_sharpe_idx = np.argmax(sharpe_array)
    max_sharpe_return = returns_array[max_sharpe_idx]
    max_sharpe_volatility = volatility_array[max_sharpe_idx]

    # Find portfolio with minimum volatility
    min_vol_idx = np.argmin(volatility_array)
    min_vol_return = returns_array[min_vol_idx]
    min_vol_volatility = volatility_array[min_vol_idx]

    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        volatility_array, returns_array, c=sharpe_array, cmap="viridis", marker="o", s=10, alpha=0.3
    )

    # Plot maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_volatility,
        max_sharpe_return,
        color="red",
        marker="*",
        s=200,
        label="Maximum Sharpe ratio",
    )

    # Plot minimum volatility portfolio
    plt.scatter(
        min_vol_volatility,
        min_vol_return,
        color="green",
        marker="*",
        s=200,
        label="Minimum volatility",
    )

    # Plot market portfolio
    plt.scatter(
        market_std, market_return, color="blue", marker="*", s=200, label="Market portfolio"
    )

    plt.colorbar(scatter, label="Sharpe ratio")
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Cumulative Return")
    plt.title("Efficient Frontier with Market Portfolio (Using Cumulative Returns)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    return {
        "Maximum Sharpe Ratio Portfolio": {
            "Return": max_sharpe_return,
            "Volatility": max_sharpe_volatility,
            "Sharpe Ratio": sharpe_array[max_sharpe_idx],
            "Weights": dict(zip(returns_df.columns, weights_array[max_sharpe_idx])),
        },
        "Minimum Volatility Portfolio": {
            "Return": min_vol_return,
            "Volatility": min_vol_volatility,
            "Sharpe Ratio": sharpe_array[min_vol_idx],
            "Weights": dict(zip(returns_df.columns, weights_array[min_vol_idx])),
        },
        "Market Portfolio": {
            "Return": market_return,
            "Volatility": market_std,
            "Sharpe Ratio": market_sharpe,
            "Weights": dict(zip(returns_df.columns, market_weights)),
        },
    }
