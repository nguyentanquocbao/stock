from visualize import MarketVisualize
from dataclasses import dataclass   , field
import pandas as pd
import logging

@dataclass
class Tracking(MarketVisualize):
    # weight:pd.DataFrame
    weight: pd.DataFrame = field(default_factory=pd.DataFrame)
    def __str__(self):
        print("Tracking by Kevin")
    def single_port_summary(self, year_build, rank,return_col,market_selection,keep_period=1):
        
        df= self.stocks_data.copy()
        stock_weight = self.weight.loc[year_build,rank].copy()
        market=self.market_data[self.market_data['ticker']==market_selection].copy()
        
        df = df[(df['ticker'].isin(stock_weight.index))
                & (df['time'].dt.year >= year_build)
                & (df['time'].dt.year <= year_build+keep_period)]
        logging.info(f"yearbuild: {year_build}")
        logging.info(f"Min time {df['time'].min()}")
        logging.info(f"Max time {df['time'].max()}")
        df["weight"] = df["ticker"].map(stock_weight)
        df=df[df['weight']>0]
        if not stock_weight.drop('year').sum().round(2) == 1:
            logging.warning(f"Sum of weights for year_build {year_build} and rank {rank} is not equal to 1: {stock_weight.sum()}")
        df['return']=df['return']*df['weight']
        df['log_return']=df['log_return']*df['weight']
        df = df[['return','log_return','time']].groupby(['time']).sum()
        df=df.merge(market[['time','log_return','return']],how='left',on='time',
                                    suffixes=('','_market'))
        def statiscal_sumary(df,return_col='return'):
            df['cumulative_return']=(df['return']+1).cumprod()
            df['cumulative_return_market']=(df['return_market']+1).cumprod()
            df['cumulative_log_return']=(df['log_return']+1).cumsum()
            df['cumulative_log_return_market']=(df['log_return_market']+1).cumsum()
            # Portfolio statistics
            # Calculate downside returns and other metrics
            period_return =  df[return_col].mean() * 252 * keep_period
            cumulative_period_return=df['cumulative_'+return_col].iloc[-1]
            period_return_market =  df[return_col+'_market'].mean() * 252 * keep_period
            cumulative_period_return_market=df['cumulative_'+return_col+'_market'].iloc[-1]
            stats_df=pd.DataFrame()
            for i in ['','cumulative_']:

                max_drawdown = (df[i+return_col].cummax() - df[i+return_col]).max()
                downside_returns = df[i+return_col][df[i+return_col] < 0]
                excess_returns = df[i+return_col] - df[i+return_col+'_market']
            # Additional portfolio statistics
                tracking_error = excess_returns.std()
                information_ratio = excess_returns.mean() / tracking_error if tracking_error != 0 else 0
                sortino_ratio = (period_return- self.risk_free_rate) / downside_returns.std() if len(downside_returns) > 0 else 0
                modified_sharpe_ratio = (period_return - self.risk_free_rate) / (df[i+return_col].std() ** 2)
                calmar_ratio = (period_return - self.risk_free_rate) / max_drawdown if max_drawdown != 0 else 0

                # Calculate portfolio and market statistics
                if i == 'cumulative_':
                    period_return_value = cumulative_period_return
                    period_return_market_value = cumulative_period_return_market
                else:
                    period_return_value = period_return
                    period_return_market_value = period_return_market

                stats_dict = {
                    'year_train': year_build,
                    'rank': rank,
                    i + 'period_return': period_return_value,
                    i + 'std': df[i + return_col].std(),
                    i + 'beta': df[i + return_col].cov(df[i + return_col + '_market']) / df[i + return_col + '_market'].var(),
                    i + 'period_return_market': period_return_market_value,
                    i + 'market_std': df[i + return_col + '_market'].std(),
                    i + 'tracking_error': tracking_error,
                    i + 'information_ratio': information_ratio,
                    i + 'sortino_ratio': sortino_ratio,
                    i + 'modified_sharpe_ratio': modified_sharpe_ratio,
                    i + 'calmar_ratio': calmar_ratio,
                    i + 'max_drawdown': max_drawdown,
                }
                
                # Add alpha, sharpe ratio, and treynor ratio
                stats_dict[i+'alpha'] = stats_dict[i+'period_return'] - stats_dict[i+'beta'] * stats_dict[i+'period_return']
                stats_dict[i+'sharpe_ratio'] = (stats_dict[i+'period_return'] - self.risk_free_rate) / stats_dict[i+'std']
                stats_dict[i+'treynor_ratio'] = (stats_dict[i+'period_return'] - self.risk_free_rate) / stats_dict[i+'beta']
                stats_dict[i+'market_sharpe_ratio'] = (stats_dict[i+'period_return'] - self.risk_free_rate) / stats_dict[i+'market_std']
                stats_dict=pd.DataFrame([stats_dict])
                stats_df = pd.concat([stats_df, stats_dict.set_index(['year_train', 'rank'])], axis=1, ignore_index=False)

                stats_df['year_train'] = year_build
                stats_df['rank'] = rank
            return stats_df
        
        # Store statistics in DataFrame attribute
        return df,statiscal_sumary(df,return_col)
    def all_port_summary(self,keep_period,return_col,market_selection='vni',):
        max_year=self.stocks_data['time'].dt.year.max()-keep_period
        available_ranks = self.weight.index.tolist()
        all_combinations = [(single[0], single[1]) for single in available_ranks if single[0]<=max_year]
        from joblib import Parallel,delayed

        # Try parallel processing, fall back to sequential if error occurs
        try:
            results = Parallel(n_jobs=-10)(
                delayed(self.single_port_summary)(single[0], single[1], return_col,market_selection, keep_period,)
                for single in all_combinations
            )
        except (ZeroDivisionError, Exception) as e:
            logging.warning(f"Parallel processing failed, falling back to sequential processing: {e}")
            results = []
            for single in all_combinations:
                result = self.single_port_summary(single[0], single[1],return_col, market_selection, keep_period)
                results.append(result)
        
        # Separate returns and stats dataframes
        returns_dfs = [result[0] for result in results]
        stats_dfs = [result[1] for result in results]
        
        # Concatenate all results
        all_returns = pd.concat(returns_dfs, keys=all_combinations)
        all_stats = pd.concat(stats_dfs, ignore_index=True)
        
        return all_returns, all_stats

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
def plot_year_attributes(df, year, attribute='all'):
    df = df[df['year_train']==year]
    
    # Define attributes to plot
    plot_attributes = [
                        'period_return', 
                        'std', 
                        'beta', 
                        'alpha',
                        'tracking_error', 
                        'information_ratio', 
                        'sortino_ratio', 
                        'sharpe_ratio', 
                        'modified_sharpe_ratio',
                        'calmar_ratio',
                        'treynor_ratio']
    # if attribute == 'all' else [attribute]
    
    # Calculate number of rows and columns for subplots
    plt.figure(figsize=(16, 8))
    n_plots = len(plot_attributes)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Plot each attribute
    for i, attr in enumerate(plot_attributes):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # Create line plot
        ax.plot(df['rank'], df[attr], marker='o')
        
        # Add labels and title
        ax.set_xlabel('Rank')
        ax.set_ylabel(attr)
        ax.set_title(f'{attr} by Rank for Year {year}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def plot_rank_attributes(df, rank, attribute='all'):
    df = df[df['rank']==rank].sort_values('year_train')
    
    # Define attributes to plot
    plot_attributes = [
                        'period_return', 
                        'std', 
                        'beta', 
                        'alpha',
                        'tracking_error', 
                        'information_ratio', 
                        'sortino_ratio', 
                        'sharpe_ratio', 
                        'modified_sharpe_ratio',
                        'calmar_ratio',
                        'treynor_ratio']
    # if attribute == 'all' else [attribute]
    
    # Calculate number of rows and columns for subplots
    plt.figure(figsize=(16, 8))
    n_plots = len(plot_attributes)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Plot each attribute
    for i, attr in enumerate(plot_attributes):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # Create line plot
        ax.plot(df['year_train'], df[attr], marker='o')
        
        # Add labels and title
        ax.set_xlabel('Rank')
        ax.set_ylabel(attr)
        ax.set_title(f'{attr} by year for rank {rank}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()