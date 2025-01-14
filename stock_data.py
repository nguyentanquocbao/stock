"""
_summary_
Module contain function to create or update data from vnstock
and store as parquet file locally
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from icecream import ic
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from vnstock3 import Vnstock as vn


def get_past_friday() -> str:
    """
    Returns the most recent Friday (including current day if it's Friday).
    """
    today = datetime.today().date()
    offset = (today.weekday() - 4) % 7
    last_friday = today - pd.Timedelta(days=offset)
    return last_friday.strftime("%Y-%m-%d")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock or indices data, remove invalid values, and calculate returns.

    """
    df = data.copy().drop_duplicates(subset=["ticker", "time"])
    df.loc[df["close"] <= 0, "close"] = None
    df.loc[df["total_outstanding"] <= 0, "total_outstanding"] = None
    df["close"] = df.groupby("ticker")["close"].ffill()
    df["total_outstanding"] = df.groupby("ticker")["total_outstanding"].ffill()
    df.dropna(subset=["close", "total_outstanding"], inplace=True)
    df["market_value"] = df["close"] * df["total_outstanding"]
    df["return"] = df.groupby("ticker")["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df.groupby("ticker")["close"].shift(1))
    if "market_value" not in df.columns:
        df["market_value"] = 1
    return df.sort_values(["ticker", "time"])


@dataclass
class StockData:
    """
    Class for updating stock data from vnstock with local parquet storage.
    """

    path: str
    path_dictionary: str

    def __post_init__(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logging.info("Created a new path for stock data")
        with open(self.path_dictionary, "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)

    def get_ticker(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the ticker list.
        If the ticker list file does not exist locally, this method creates it
        by reloading the data and then reads from a parquet file. The resulting
        DataFrame is returned for further processing or analysis.
        """
        if not os.path.exists(self.path + self.dictionary["path_ticker_list"]):
            logging.info("Creating data for the first time")
            self.reload_ticker_list()
        return pd.read_parquet(self.path + self.dictionary["path_ticker_list"])

    def get_indices_data(self) -> List[pd.DataFrame]:
        """
        Fetch and return cleaned historical data for the VNINDEX and VN30 indices.
        This method retrieves the historical prices for the specified indices from a
        pre-configured data source, cleans the returned data, and returns it as a list
        of DataFrames.
        Returns:
            List[pd.DataFrame]: A list containing two cleaned DataFrames, one for each index.
        """
        stock = vn(show_log=False).stock(symbol="ABC", source=self.dictionary["source"])
        vni = stock.quote.history(
            symbol="VNINDEX", start="2013-01-01", end=get_past_friday(), interval="1D"
        )
        vni["ticker"], vni["exchange"], vni["total_outstanding"] = "vni", "vni", 1
        vn30 = stock.quote.history(
            symbol="VN30", start="2013-01-01", end=get_past_friday(), interval="1D"
        )
        vn30["ticker"], vn30["exchange"], vn30["total_outstanding"] = "vn30", "vn30", 1
        return [
            clean_data(vni)[
                [
                    "exchange",
                    "time",
                    "high",
                    "low",
                    "return",
                    "close",
                    "log_return",
                    "volume",
                    "market_value"
                ]
            ],
            clean_data(vn30)[
                [
                    "exchange",
                    "time",
                    "high",
                    "low",
                    "return",
                    "log_return",
                    "close",
                    "volume",
                    "market_value"
                ]
            ],
        ]

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def read_outstanding_1_stock(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reads the outstanding shares for a specified stock ticker from the provided DataFrame.
        This function filters the given DataFrame by the specified ticker and attempts to update
        the 'total_outstanding' column with the share outstanding value fetched from a price board.
        If the share outstanding data is not available, the function changes the stock's exchange
        to 'DELISTED'. If a connection error occurs, it raises that exception.
        Args:
            ticker (str): The stock ticker symbol to filter the DataFrame.
            data (pd.DataFrame): The DataFrame that contains stock data.
        Returns:
            pd.DataFrame: The filtered copy of the DataFrame with an updated 'total_outstanding'
                column if available, or with the 'exchange' set to 'DELISTED' otherwise.
        """

        data = data[data[self.dictionary["ticker"]] == ticker]
        data_copy = data.copy()
        stock = vn(show_log=False).stock(symbol="ABC", source=self.dictionary["source"])
        try:
            data_copy["total_outstanding"] = stock.trading.price_board([ticker])[
                self.dictionary["share_outstanding"][0]
            ][self.dictionary["share_outstanding"][1]][0]
        except (KeyError, IndexError):
            data_copy.loc[
                (data_copy[self.dictionary["ticker"]] == ticker)
                & (data_copy["exchange"] != "BOND"),
                "exchange",
            ] = "DELISTED"
        except ConnectionError:
            logging.warning("Connection error on %s", ticker)
            raise
        return data_copy

    def reload_ticker_list(self) -> None:
        """
        Reloads the list of tickers by fetching data from a predefined source, then merges and
        stores the results.
        This method performs the following steps:
        1. Retrieves a list of stock symbols from the specified data source.
        2. Uses parallel processing to read outstanding information for each stock.
        3. Concatenates the collected data into a single DataFrame and assigns the current date.
        4. Checks if a previous ticker list exists:
            - If not, writes the DataFrame to a parquet file.
            - If it exists, merges the new data with the old data based on the creation date,
            then updates the existing file.
        Returns:
            None: This method does not return anything. It updates a parquet file that stores
            the ticker information.
        Raises:
            Any exceptions raised by reading or writing parquet files, or parallel processing
            errors will propagate to the caller.
        """

        stock = vn(show_log=False).stock(symbol="ABC", source=self.dictionary["source"])
        new_ticker = stock.listing.symbols_by_exchange()
        all_ticker = Parallel(n_jobs=-2)(
            delayed(self.read_outstanding_1_stock)(stk, new_ticker)
            for stk in new_ticker[self.dictionary["ticker"]].unique()
        )
        all_ticker = pd.concat(all_ticker)
        all_ticker["created_time"] = datetime.today().date().strftime("%Y-%m-%d")
        path_ticker_list = self.path + self.dictionary["path_ticker_list"]
        if not os.path.exists(path_ticker_list):
            all_ticker.to_parquet(path_ticker_list)
        else:
            old_ticker = pd.read_parquet(path_ticker_list)
            self.concat_new_and_old(
                old_ticker, all_ticker, "created_time", self.dictionary["path_ticker_list"]
            )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(ConnectionError),
        after=lambda r: logging.warning("Retry attempt %s for data retrieval", r.attempt_number),
    )
    def read_1_stock_data(
        self, ticker: str, end_date: str, start_date: str = "2013-01-01"
    ) -> pd.DataFrame:
        """
        Fetches historical stock data for a single ticker symbol.
        This method retrieves daily stock price data from a specified data source
        for a given stock ticker between start and end dates.
        Parameters
        ----------
        ticker : str
            The stock ticker symbol
        end_date : str
            The end date for data retrieval in 'YYYY-MM-DD' format
        start_date : str, optional
            The start date for data retrieval in 'YYYY-MM-DD' format, defaults to '2013-01-01'
        Returns
        -------
        pd.DataFrame
            DataFrame containing daily stock data with ticker column added.
            Returns empty DataFrame if data retrieval fails.
        Notes
        -----
        Uses vnstock library to fetch data from configured source.
        Data interval is fixed at daily ('1D').
        """
        try:
            data = (
                vn(show_log=False)
                .stock(symbol=ticker, source=self.dictionary["source"])
                .quote.history(start=start_date, end=end_date, interval="1D")
            )
            data["ticker"] = ticker
            return data
        except (ValueError, TypeError, KeyError):
            return pd.DataFrame()

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieves and processes stock market data.

        This method fetches stock data either from an existing parquet file or downloads it if not available.
        If existing data is outdated (older than last Friday), it updates the dataset.

        Returns:
            pd.DataFrame: A cleaned DataFrame containing merged stock data with the following:
                - Basic stock data (from read_1_stock_data)
                - Exchange information
                - Total outstanding shares
                - Other ticker information

        Notes:
            - Uses parallel processing to fetch individual stock data
            - Stores data in parquet format for faster future access
            - Automatically updates if data is older than the most recent Friday
            - Merges basic stock data with additional ticker information
            - Cleans the final dataset before returning

        Raises:
            Any exceptions from underlying methods (read_1_stock_data, clean_data) will propagate
        """
        ticker = self.get_ticker()
        path_temp = self.path + self.dictionary["path_ticker_daily_data"]
        if os.path.exists(path_temp):
            data = pd.read_parquet(path_temp)
            latest_data_date = data["time"].max()
            if latest_data_date.strftime("%Y-%m-%d") < get_past_friday():
                logging.info("Updating data")
                updated_ticker_data = pd.concat(
                    Parallel(n_jobs=-15)(
                        delayed(self.read_1_stock_data)(stk, get_past_friday())
                        for stk in ticker[self.dictionary["ticker"]].unique()
                    )
                )
                updated_ticker_data.to_parquet(path_temp)
                all_ticker = updated_ticker_data
            else:
                all_ticker = data.copy()
        else:
            logging.info("Downloading entire data set")
            all_ticker = pd.concat(
                Parallel(n_jobs=-15)(
                    delayed(self.read_1_stock_data)(stk, get_past_friday())
                    for stk in ticker[self.dictionary["ticker"]].unique()
                )
            )
            all_ticker.to_parquet(path_temp)

        ticker_subset = ticker[[self.dictionary["ticker"], "exchange", "total_outstanding"]].copy()
        merged_data = all_ticker.merge(
            ticker_subset, how="inner", left_on=["ticker"], right_on=[self.dictionary["ticker"]]
        ).copy()
        merged_data = clean_data(merged_data)
        logging.info("Max time in merged data: %s", merged_data["time"].max())
        summary = merged_data.groupby("exchange")["ticker"].nunique()
        logging.info("Summary of tickers by exchange: %s", summary.to_dict())
        return merged_data, self.compute_market_level(merged_data)

    def concat_new_and_old(
        self, old: pd.DataFrame, new: pd.DataFrame, date_col: str, path_out: str
    ):
        """
        Concatenates old and new DataFrames, sorts by date, removes duplicates, and saves if new data exists.
        This method combines existing data with new data, ensuring proper date ordering and
        handling of duplicate entries. If the combined dataset is larger than the original,
        it saves the updated dataset to parquet format.
        Parameters
        ----------
        old : pd.DataFrame
            Existing DataFrame containing historical data
        new : pd.DataFrame
            New DataFrame containing data to be added
        date_col : str
            Name of the column containing date information
        path_out : str
            Output path where the combined parquet file will be saved
        Returns
        -------
        None
            The method saves the combined DataFrame to disk if new data exists
        """
        combined = pd.concat([old, new], axis=0)
        combined.sort_values(by=date_col, ascending=True, inplace=True)
        subset_cols = [c for c in combined.columns if c != date_col]
        combined.drop_duplicates(subset=subset_cols, keep="first", inplace=True)
        if combined.size > old.size:
            logging.info("There is new data; updated.")
            combined.to_parquet(self.path + path_out)

    def compute_market_level(self, all_stock_data) -> pd.DataFrame:
        """
        Compute the market level returns and log returns for each exchange.
        This method aggregates the weighted returns and log returns for each exchange
        and computes the overall market level returns and log returns.
        Returns:
        pd.DataFrame: A DataFrame containing the market level returns and log returns
        for each exchange.
        """
        indies_data = self.get_indices_data()
        all_stock_data = pd.concat([all_stock_data] + indies_data, axis=0)
        all_stock_data["market_weight"] = all_stock_data["market_value"] / all_stock_data.groupby(
            ["exchange", "time"]
        )["market_value"].transform("sum")

        cols_to_weight = ["log_return", "return", "high", "low", "close", "volume"]

        for col in cols_to_weight:
            all_stock_data[col] = (
                all_stock_data["market_weight"] * all_stock_data[col]
            )

        market_data = (
            all_stock_data.groupby(["exchange", "time"])[cols_to_weight].sum().reset_index()
        )
        market_data['year']=market_data['time'].dt.year
        market_data['ticker'],market_data['exchange']=market_data['exchange'],'index'
        return market_data


