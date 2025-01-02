"""
A Python module for backtesting and analyzing portfolio strategies using efficient frontier optimization.

This module provides functionality for:
- Transforming stock market data into return matrices
- Building efficient portfolios using optimization
- Tracking portfolio performance over time periods
- Visualizing backtest results
- Performing statistical tests on portfolio performance

The module contains the following main functions:

- transform_df: Transforms stock data DataFrame into return matrix
- build_port: Calculates efficient frontier using optimization
- tracking: Performs year-by-year portfolio performance tracking
- plot_backtest_stategy: Visualizes backtest results across portfolio ranks
- perform_portfolio_tests: Conducts statistical tests on portfolio results

The module requires the following dependencies:
- pandas
- numpy 
- matplotlib
- seaborn
- scipy
- joblib
- IPython

Key features:
- Handles missing data through forward/backward filling
- Parallel processing for portfolio optimization
- Comprehensive statistical testing
- Visualization of results through multiple plots
- Capital Allocation Line (CAL) portfolio analysis

"""

import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from icecream import ic
from efficient_frontier import portfolio_statistics, calculate_efficient_frontier_points


def transform_df(df):
    """Transform a DataFrame containing stock data into return matrix.
    This function processes a DataFrame containing stock market data
        to create a matrix of log returns,
    where stocks with missing market values are removed from the analysis.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns:
            - time: timestamp for the observation
            - ticker: stock identifier
            - market_value: market value of the stock
            - log_return: logarithmic returns of the stock
    Returns
    -------
    pandas.DataFrame
        A pivot table with:
            - Index: time stamps
            - Columns: stock tickers
            - Values: log returns
        Only includes stocks that have complete market value
            data after forward and backward filling.
    Notes
    -----
    The function performs the following steps:
    1. Creates a pivot table of market values
    2. Fills missing values using backward and forward fill
    3. Removes stocks with any remaining missing values
    4. Normalizes weights
    5. Returns a pivot table of log returns for the remaining stocks
    """

    weights = df.pivot_table(index="time", values="market_value", columns="ticker")
    weights = weights.bfill()
    weights = weights.ffill()
    # weights = weights.fillna(method="ffill")
    weights.dropna(axis=1, inplace=True)
    weights = weights.div(weights.sum(axis=0), axis=1)

    return df[df["ticker"].isin(weights.columns)].pivot_table(
        index="time", values="log_return", columns="ticker"
    )


def build_port(
    returns: pd.DataFrame,
    num_portfolios: int = 12,
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

    # Porfolio computation
    returns = (1 + returns).cumprod()
    # Constraints
    num_assets = len(returns.columns)
    bounds = tuple(range0 for _ in range(num_assets))  # weights between given range
    # Initial guess (equal weights)
    init_weights = np.array([1 / num_assets] * num_assets)
    sorted_unique_returns = returns.iloc[-1].dropna().sort_values().unique()
    # Determine second minimum and second maximum
    if len(sorted_unique_returns) >= 2:
        second_min, second_max = sorted_unique_returns[1], sorted_unique_returns[-2]
    elif len(sorted_unique_returns) == 1:
        second_min, second_max = sorted_unique_returns[0], sorted_unique_returns[0]
    else:
        raise ValueError("Not enough return data to calculate second minimum and maximum.")

    # Generate target returns using the second minimum and second maximum
    target_returns = np.linspace(second_min, second_max, num_portfolios)

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
                return_val, std_val, sharpe_val, _ = portfolio_statistics(result.x, returns)
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

    if len(results_array.shape) < 2:
        print("Invalid results array shape!")
        return None, None

    return results_array


def tracking(
    df: pd.DataFrame,
    stock_list: List[str],
    num_portfolios=10,
):
    """
    Calculates the tracking performance of portfolios over time using training and testing datasets.
    This function performs a year-by-year analysis of portfolio performance by:
    1. Splitting data into training (year i) and testing (year i+1) sets
    2. Building portfolios using training data
    3. Evaluating portfolio performance on both training and testing periods
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at minimum the following columns:
        - time: datetime column with the time periods
        - market_value: numeric column with market values
        - ticker: string column with stock tickers
    stock_list : list-like
        List of stock tickers to consider in the analysis
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing tracking results with columns:
        - train_return: Portfolio returns on training data
        - train_std: Standard deviation on training data
        - test_return: Portfolio returns on testing data
        - test_std: Standard deviation on testing data
        - year: The training year
    Notes
    -----
    - Only considers stocks with 3-character tickers and positive market values
    - Requires transform_df() and build_port() helper functions
    - Skips the last year in the dataset as it cannot be used for testing
    """

    all_tracking = []
    all_track_df = []
    for i in df["time"].dt.year.unique():
        if i < df["time"].dt.year.max():
            train = df[
                (df["market_value"] > 0)
                & (df["time"].dt.year == i)
                & (df["ticker"].str.len() == 3)
                & (df["ticker"].isin(stock_list))
            ]
            test = df[
                (df["market_value"] > 0)
                & (df["time"].dt.year == i + 1)
                & (df["ticker"].str.len() == 3)
                & (df["ticker"].isin(stock_list))
            ]
            test = transform_df(test[test["ticker"].isin(train["ticker"].unique())])
            train = transform_df(train)
            result = build_port(train, num_portfolios=num_portfolios + 2, range0=(0, 1))
            for port_no, weight in enumerate(result):
                if weight is not None:
                    train_out = portfolio_statistics(weight[3:], (1 + train).cumprod())
                    test_out = portfolio_statistics(weight[3:], (1 + test).cumprod())
                    track_df = pd.concat([train_out[3], test_out[3]], axis=1, ignore_index=False)
                    tracking_result = pd.DataFrame(
                        {
                            "train_return": [train_out[0]],
                            "train_std": [train_out[1]],
                            "test_return": [test_out[0]],
                            "test_std": [test_out[1]],
                        }
                    )
                    track_df.columns = ["train", "test"]
                    tracking_result["year"] = [i]
                    track_df["year"] = i
                    track_df["port"] = port_no
                    all_tracking.append(tracking_result)
                    all_track_df.append(track_df)
    all_tracking = pd.concat(all_tracking, axis=0)
    all_tracking = all_tracking.sort_values(["year", "train_return"])
    all_tracking["rank"] = all_tracking.groupby("year").cumcount() + 1

    year_list = [i["year"].unique()[0] for i in all_track_df]
    df0 = [
        pd.concat([i.reset_index() for i in all_track_df if i["year"].unique()[0] == year], axis=0)
        for year in set(year_list)
    ]
    all_tracking["CAL"] = False
    for i in df["time"].dt.year.unique():
        temp = all_tracking.loc[all_tracking["year"] == i]
        frontier_x, frontier_y = calculate_efficient_frontier_points(
            np.column_stack((temp["train_std"], temp["train_return"])),
            0.05,
        )
        all_tracking.loc[
            (all_tracking["train_return"] == frontier_y[-1])
            & (all_tracking["year"] == i)
            & (all_tracking["train_std"] == frontier_x[-1]),
            "CAL",
        ] = True

    return all_tracking.sort_values(["year", "train_return"]), df0


def plot_backtest_stategy(df, market_df_stat, num_portfolios=10):
    """Plots backtest strategy results for different portfolio ranks and CAL portfolio.
    This function creates a visualization of portfolio performance metrics across different ranks
    and for the Capital Allocation Line (CAL) portfolio. It generates multiple subplots showing
    the training and testing returns as lines, and their corresponding standard deviations as bars.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the backtest results with columns:
        - 'year': The year of the observation
        - 'rank': Portfolio rank (1-10)
        - 'CAL': Boolean indicating if the portfolio is CAL
        - 'train_return : Training set returns
        - 'test_return': Testing set returns
        - 'train_std': Training set standard deviation
        - 'test_std': Testing set standard deviation
    Returns
    -------
    None
        Displays a matplotlib figure with 11 subplots:
        - 10 subplots for each rank (1-10)
        - 1 subplot for CAL portfolio performance
    Notes
    -----
    The function uses a 5x2 subplot layout with consistent y-axis scaling across all plots
    for better comparison. Each subplot contains:
    - Line plots for returns (training and testing)
    - Bar plots for standard deviations (training and testing)
    - Dual y-axes for returns and standard deviations
    - Legend indicating all metrics
    """

    _ = plt.figure(figsize=(20, 25))
    # Create an additional subplot for CAL=True data

    # First calculate the overall min and max values for consistent scaling
    y1_min = min(df["train_return"].min(), df["test_return"].min())
    y1_max = max(df["train_return"].max(), df["test_return"].max())
    y2_min = min(df["train_std"].min(), df["test_std"].min())
    y2_max = max(df["train_std"].max(), df["test_std"].max())

    # Plot each rank in a separate subplot
    for i in range(1, num_portfolios + 1):
        mask = df["rank"] == i
        ax = plt.subplot(int(num_portfolios / 2 + 1), 2, i)

        # Plot returns as lines
        ax.plot(df[mask]["year"], df[mask]["train_return"], "b-", label="Train Return")
        ax.plot(df[mask]["year"], df[mask]["test_return"], "g-", label="Test Return")
        ax.plot(
            market_df_stat["year"], market_df_stat["market_log_return"], "r-", label="Market Return"
        )
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)  # Add horizontal line at y=1
        ax.set_ylim(y1_min * 0.9, y1_max * 1.1)  # Set consistent y-axis range for returns

        # Create second y-axis for standard deviations
        ax2 = ax.twinx()
        ax2.bar(
            df[mask]["year"] - 0.2,
            df[mask]["train_std"],
            0.4,
            color="lightblue",
            alpha=0.5,
            label="Train Std",
        )
        ax2.bar(
            df[mask]["year"] + 0.2,
            df[mask]["test_std"],
            0.4,
            color="lightgreen",
            alpha=0.5,
            label="Test Std",
        )
        ax2.bar(
            market_df_stat["year"],
            market_df_stat["std"],
            0.4,
            color="red",
            alpha=0.5,
            label="market std",
        )
        ax2.set_ylim(y2_min * 0.9, y2_max * 1.1)  # Set consistent y-axis range for std

        # Set labels and title
        ax.set_xlabel("Year")
        ax.set_ylabel("Returns", color="b")
        ax2.set_ylabel("Standard Deviation", color="b")
        plt.title(f"Rank {i} Portfolio Performance")

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
    ax_cal = plt.subplot(int(num_portfolios / 2 + 1), 2, num_portfolios + 1)  # Add one more subplot
    cal_mask = df["CAL"] == True

    # Plot returns as lines for CAL=True data
    ax_cal.plot(
        df[cal_mask]["year"], df[cal_mask]["train_return"], "b-", label="Train Return (CAL)"
    )
    ax_cal.plot(df[cal_mask]["year"], df[cal_mask]["test_return"], "g-", label="Test Return (CAL)")
    ax_cal.plot(
        market_df_stat["year"], market_df_stat["market_log_return"], "r-", label="Market Return"
    )
    ax_cal.axhline(y=1, color="gray", linestyle="--", alpha=0.5)  # Add horizontal line at y=1
    ax_cal.set_ylim(y1_min * 0.9, y1_max * 1.1)

    # Create second y-axis for standard deviations
    ax_cal2 = ax_cal.twinx()
    ax_cal2.bar(
        df[cal_mask]["year"] - 0.2,
        df[cal_mask]["train_std"],
        0.4,
        color="lightblue",
        alpha=0.5,
        label="Train Std (CAL)",
    )
    ax_cal2.bar(
        df[cal_mask]["year"] + 0.2,
        df[cal_mask]["test_std"],
        0.4,
        color="lightgreen",
        alpha=0.5,
        label="Test Std (CAL)",
    )
    ax_cal2.bar(
        market_df_stat["year"],
        market_df_stat["std"],
        0.4,
        color="red",
        alpha=0.5,
        label="market std",
    )
    ax_cal2.set_ylim(y2_min * 0.9, y2_max * 1.1)

    # Set labels and title
    ax_cal.set_xlabel("Year")
    ax_cal.set_ylabel("Returns", color="b")
    ax_cal2.set_ylabel("Standard Deviation", color="r")
    plt.title("CAL Portfolio Performance")

    # Add legend
    lines_cal1, labels_cal1 = ax_cal.get_legend_handles_labels()
    lines_cal2, labels_cal2 = ax_cal2.get_legend_handles_labels()
    ax_cal.legend(lines_cal1 + lines_cal2, labels_cal1 + labels_cal2, loc="upper left")


def perform_portfolio_tests(results):
    """
    Perform statistical tests comparing train and test periods for returns and volatility.

    Parameters:
    results (pd.DataFrame): DataFrame containing train and test results

    Returns:
    dict: Dictionary containing test results and conclusions
    """
    tests_results = {}

    test_configs = [
        {
            "name": "Returns (Test ≤ Train)",
            "data": (results["test_return"], results["train_return"]),
            "alternative": "less",
            "hypothesis": "H₀: test_return ≤ train_return",
            "category": "Returns",
        },
        {
            "name": "Volatility (Test ≥ Train)",
            "data": (results["test_std"], results["train_std"]),
            "alternative": "greater",
            "hypothesis": "H₀: test_std ≥ train_std",
            "category": "Volatility",
        },
        {
            "name": "Returns (Test > Train)",
            "data": (results["test_return"], results["train_return"]),
            "alternative": "greater",
            "hypothesis": "H₀: test_return > train_return",
            "category": "Returns",
        },
        {
            "name": "Volatility (Test < Train)",
            "data": (results["test_std"], results["train_std"]),
            "alternative": "less",
            "hypothesis": "H₀: test_std < train_std",
            "category": "Volatility",
        },
        {
            "name": "Returns Equality",
            "data": (results["test_return"], results["train_return"]),
            "alternative": "two-sided",
            "hypothesis": "H₀: test_return = train_return",
            "category": "Returns",
        },
        {
            "name": "Volatility Equality",
            "data": (results["test_std"], results["train_std"]),
            "alternative": "two-sided",
            "hypothesis": "H₀: test_std = train_std",
            "category": "Volatility",
        },
    ]

    for config in test_configs:
        t_stat, p_val = stats.ttest_ind(*config["data"], alternative=config["alternative"])
        tests_results[config["name"]] = {
            "category": config["category"],
            "hypothesis": config["hypothesis"],
            "t_statistic": round(float(t_stat), 4),  # type: ignore
            "p_value": round(float(p_val), 4),  # type: ignore
            "conclusion": "Reject" if p_val < 0.05 else "Fail to reject",  # type: ignore
            "significance": "Significant" if p_val < 0.05 else "Not significant",  # type: ignore
        }

    # Convert results to DataFrame for better presentation
    df_results = pd.DataFrame(tests_results).T

    # Create visualizations
    plt.figure(figsize=(12, 6))

    # P-value plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_results.reset_index(), x="index", y="p_value")
    plt.axhline(y=0.05, color="r", linestyle="--", label="5% significance level")
    plt.xticks(rotation=45)
    plt.title("P-values by Test")
    plt.legend()

    # T-statistic plot
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_results.reset_index(), x="index", y="t_statistic")
    plt.xticks(rotation=45)
    plt.title("T-statistics by Test")

    plt.tight_layout()

    # Style the DataFrame
    styled_df = (
        df_results.style.apply(
            lambda x: [
                "background-color: #90EE90" if v < 0.05 else "background-color: #FFB6C1" for v in x
            ],
            subset=["p_value"],
        )
        .format({"t_statistic": "{:.4f}", "p_value": "{:.4f}"})
        .set_properties(**{"text-align": "center"})  # type: ignore
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("background-color", "#f0f0f0")],
                },
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )

    # Display results
    print("\n=== Portfolio Statistical Tests Results ===\n")
    display(styled_df)
    plt.show()

    return tests_results
