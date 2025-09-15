#!/usr/bin/env python3
"""
S&P 500 Log Returns Data Preparation for DistVAE

This script prepares S&P 500 data for modeling with the standalone DistVAE.
It calculates log returns and creates features suitable for distributional learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load S&P 500 datasets and perform initial exploration"""
    print("Loading S&P 500 datasets...")

    # Load data
    stocks = pd.read_csv('data/sp500/sp500_stocks.csv')
    companies = pd.read_csv('data/sp500/sp500_companies.csv')
    index = pd.read_csv('data/sp500/sp500_index.csv')

    print(f"Raw data shapes:")
    print(f"  Stocks: {stocks.shape}")
    print(f"  Companies: {companies.shape}")
    print(f"  Index: {index.shape}")

    return stocks, companies, index

def clean_stocks_data(stocks, companies, min_observations=252*2):
    """
    Clean stocks data and filter for quality stocks

    Args:
        stocks: Raw stocks DataFrame
        companies: Companies metadata
        min_observations: Minimum trading days required (default: 2 years)
    """
    print("Cleaning stocks data...")

    # Convert date to datetime
    stocks['Date'] = pd.to_datetime(stocks['Date'])

    # Remove rows with missing price data
    stocks_clean = stocks.dropna(subset=['Close', 'Volume']).copy()
    print(f"After removing missing prices: {stocks_clean.shape[0]:,} observations")

    # Filter stocks with sufficient data
    stock_counts = stocks_clean.groupby('Symbol').size()
    valid_stocks = stock_counts[stock_counts >= min_observations].index
    stocks_clean = stocks_clean[stocks_clean['Symbol'].isin(valid_stocks)].copy()

    print(f"Stocks with ≥{min_observations} observations: {len(valid_stocks)}")
    print(f"Final clean dataset: {stocks_clean.shape[0]:,} observations")

    # Sort by symbol and date for proper log return calculation
    stocks_clean = stocks_clean.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    return stocks_clean, valid_stocks

def calculate_log_returns(stocks_clean):
    """Calculate log returns and additional financial features"""
    print("Calculating log returns and features...")

    # Calculate log returns (ln(P_t / P_{t-1}))
    stocks_clean['log_return'] = stocks_clean.groupby('Symbol')['Close'].apply(
        lambda x: np.log(x / x.shift(1))
    ).values

    # Calculate additional features
    stocks_clean['high_low_ratio'] = np.log(stocks_clean['High'] / stocks_clean['Low'])
    stocks_clean['volume_log'] = np.log(stocks_clean['Volume'] + 1)

    # Intraday price movements
    stocks_clean['open_close_return'] = np.log(stocks_clean['Close'] / stocks_clean['Open'])
    stocks_clean['high_open_ratio'] = np.log(stocks_clean['High'] / stocks_clean['Open'])
    stocks_clean['low_open_ratio'] = np.log(stocks_clean['Low'] / stocks_clean['Open'])

    # Remove first day for each stock (no prior price for log return)
    stocks_clean = stocks_clean.dropna(subset=['log_return']).copy()

    print(f"Dataset after adding features: {stocks_clean.shape}")

    return stocks_clean

def create_cross_sectional_features(stocks_clean, companies):
    """Create cross-sectional features across stocks for each date"""
    print("Creating cross-sectional features...")

    # Add sector information
    sector_map = companies.set_index('Symbol')['Sector'].to_dict()
    stocks_clean['Sector'] = stocks_clean['Symbol'].map(sector_map)

    # Calculate daily cross-sectional statistics
    daily_stats = stocks_clean.groupby('Date').agg({
        'log_return': ['mean', 'std', 'skew', 'min', 'max'],
        'volume_log': ['mean', 'std'],
        'high_low_ratio': ['mean', 'std']
    }).round(6)

    # Flatten column names
    daily_stats.columns = [f"{col[0]}_{col[1]}" for col in daily_stats.columns]
    daily_stats = daily_stats.reset_index()

    # Merge back to individual stock data
    stocks_with_market = stocks_clean.merge(daily_stats, on='Date', suffixes=('', '_market'))

    # Calculate relative performance vs market
    stocks_with_market['excess_return'] = (
        stocks_with_market['log_return'] - stocks_with_market['log_return_mean']
    )

    print(f"Final dataset with market features: {stocks_with_market.shape}")

    return stocks_with_market, daily_stats

def prepare_index_data(index):
    """Prepare S&P 500 index data with log returns"""
    print("Preparing S&P 500 index data...")

    index_clean = index.copy()
    index_clean['Date'] = pd.to_datetime(index_clean['Date'])
    index_clean = index_clean.sort_values('Date').reset_index(drop=True)

    # Calculate index log returns
    index_clean['index_log_return'] = np.log(
        index_clean['S&P500'] / index_clean['S&P500'].shift(1)
    )

    # Remove first observation (no prior price)
    index_clean = index_clean.dropna().reset_index(drop=True)

    print(f"Index data shape: {index_clean.shape}")

    return index_clean

def create_distvae_datasets(stocks_with_market, companies, target_date='2020-01-01'):
    """
    Create datasets optimized for DistVAE training

    Two approaches:
    1. Time-series approach: Daily cross-sectional log returns
    2. Stock-level approach: Individual stock features over time windows
    """
    print("Creating DistVAE-ready datasets...")

    # Approach 1: Daily Cross-Sectional Log Returns
    # Each row is a trading day, columns are log returns of different stocks
    print("\n=== Approach 1: Cross-Sectional Daily Returns ===")

    # Get most liquid stocks (top by trading frequency)
    stock_freq = stocks_with_market.groupby('Symbol').size().sort_values(ascending=False)
    top_stocks = stock_freq.head(50).index  # Top 50 most liquid stocks

    # Create wide format: dates x stock returns
    pivot_returns = stocks_with_market[stocks_with_market['Symbol'].isin(top_stocks)].pivot(
        index='Date', columns='Symbol', values='log_return'
    )

    # Add market-level features
    market_features = stocks_with_market.groupby('Date')[
        ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']
    ].first()

    cross_sectional_df = pd.concat([pivot_returns, market_features], axis=1)
    cross_sectional_df = cross_sectional_df.dropna()

    print(f"Cross-sectional dataset shape: {cross_sectional_df.shape}")
    print(f"Date range: {cross_sectional_df.index.min().date()} to {cross_sectional_df.index.max().date()}")

    # Split train/test
    train_cutoff = pd.to_datetime(target_date)
    train_cs = cross_sectional_df[cross_sectional_df.index < train_cutoff]
    test_cs = cross_sectional_df[cross_sectional_df.index >= train_cutoff]

    print(f"Train set: {train_cs.shape}, Test set: {test_cs.shape}")

    # Approach 2: Individual Stock Features with Rolling Statistics
    print("\n=== Approach 2: Individual Stock Features ===")

    # Calculate rolling statistics for each stock
    def add_rolling_features(group, windows=[5, 20, 60]):
        """Add rolling statistics for a stock"""
        for window in windows:
            group[f'return_mean_{window}d'] = group['log_return'].rolling(window).mean()
            group[f'return_std_{window}d'] = group['log_return'].rolling(window).std()
            group[f'volume_mean_{window}d'] = group['volume_log'].rolling(window).mean()
        return group

    stocks_rolling = stocks_with_market.groupby('Symbol').apply(add_rolling_features)
    stocks_rolling = stocks_rolling.reset_index(drop=True)
    stocks_rolling = stocks_rolling.dropna()

    # Select features for stock-level modeling
    stock_features = [
        'log_return', 'excess_return', 'high_low_ratio', 'volume_log',
        'return_mean_5d', 'return_std_5d', 'return_mean_20d', 'return_std_20d',
        'volume_mean_5d', 'volume_mean_20d',
        'log_return_mean', 'log_return_std'  # Market features
    ]

    stock_level_df = stocks_rolling[['Date', 'Symbol', 'Sector'] + stock_features].copy()

    # Encode sector as categorical
    sector_dummies = pd.get_dummies(stock_level_df['Sector'], prefix='sector')
    stock_level_df = pd.concat([
        stock_level_df.drop(['Sector'], axis=1),
        sector_dummies
    ], axis=1)

    print(f"Stock-level dataset shape: {stock_level_df.shape}")

    # Split train/test for stock-level data
    train_stock = stock_level_df[stock_level_df['Date'] < target_date].drop(['Date', 'Symbol'], axis=1)
    test_stock = stock_level_df[stock_level_df['Date'] >= target_date].drop(['Date', 'Symbol'], axis=1)

    print(f"Stock-level train: {train_stock.shape}, test: {test_stock.shape}")

    return {
        'cross_sectional': {
            'full': cross_sectional_df,
            'train': train_cs,
            'test': test_cs,
            'stock_names': list(top_stocks)
        },
        'stock_level': {
            'full': stock_level_df,
            'train': train_stock,
            'test': test_stock,
            'features': stock_features + list(sector_dummies.columns)
        }
    }

def save_datasets(datasets, index_data):
    """Save prepared datasets for DistVAE modeling"""
    print("\nSaving prepared datasets...")

    # Save cross-sectional datasets
    datasets['cross_sectional']['full'].to_csv('data/sp500/sp500_cross_sectional_full.csv')
    datasets['cross_sectional']['train'].to_csv('data/sp500/sp500_cross_sectional_train.csv')
    datasets['cross_sectional']['test'].to_csv('data/sp500/sp500_cross_sectional_test.csv')

    # Save stock-level datasets
    datasets['stock_level']['full'].to_csv('data/sp500/sp500_stock_level_full.csv', index=False)
    datasets['stock_level']['train'].to_csv('data/sp500/sp500_stock_level_train.csv', index=False)
    datasets['stock_level']['test'].to_csv('data/sp500/sp500_stock_level_test.csv', index=False)

    # Save index data
    index_data.to_csv('data/sp500/sp500_index_clean.csv', index=False)

    # Save metadata
    metadata = {
        'cross_sectional_stocks': datasets['cross_sectional']['stock_names'],
        'stock_level_features': datasets['stock_level']['features'],
        'preparation_date': datetime.now().isoformat(),
        'train_test_split': '2020-01-01'
    }

    import json
    with open('data/sp500/datasets_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Datasets saved to data/sp500/")

def generate_summary_statistics(datasets, index_data):
    """Generate summary statistics and visualizations"""
    print("\nGenerating summary statistics...")

    # Cross-sectional statistics
    cs_stats = datasets['cross_sectional']['full'].describe()
    print("\n=== Cross-Sectional Dataset Statistics ===")
    print(cs_stats)

    # Stock-level statistics
    stock_stats = datasets['stock_level']['train'].describe()
    print("\n=== Stock-Level Dataset Statistics ===")
    print(stock_stats)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Distribution of daily log returns
    all_returns = datasets['cross_sectional']['full'].iloc[:, :-4].values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    axes[0,0].hist(all_returns, bins=100, alpha=0.7, density=True)
    axes[0,0].set_title('Distribution of Daily Log Returns (All Stocks)')
    axes[0,0].set_xlabel('Log Return')
    axes[0,0].set_ylabel('Density')

    # Plot 2: Time series of market volatility
    market_vol = datasets['cross_sectional']['full']['log_return_std']
    axes[0,1].plot(market_vol.index, market_vol)
    axes[0,1].set_title('Market Volatility Over Time')
    axes[0,1].set_xlabel('Date')
    axes[0,1].set_ylabel('Cross-Sectional Std Dev')

    # Plot 3: Correlation heatmap of top 10 stocks
    top_10_stocks = datasets['cross_sectional']['stock_names'][:10]
    corr_matrix = datasets['cross_sectional']['full'][top_10_stocks].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[1,0], fmt='.2f')
    axes[1,0].set_title('Top 10 Stocks Correlation Matrix')

    # Plot 4: S&P 500 index with returns
    axes[1,1].plot(index_data['Date'], index_data['S&P500'])
    axes[1,1].set_title('S&P 500 Index Level')
    axes[1,1].set_xlabel('Date')
    axes[1,1].set_ylabel('Index Level')

    plt.tight_layout()
    plt.savefig('data/sp500/sp500_data_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Summary plots saved to data/sp500/sp500_data_summary.png")

def main():
    """Main execution function"""
    print("=== S&P 500 Data Preparation for DistVAE ===")
    print(f"Start time: {datetime.now()}")

    # Load and explore data
    stocks, companies, index = load_and_explore_data()

    # Clean stocks data
    stocks_clean, valid_stocks = clean_stocks_data(stocks, companies)

    # Calculate log returns and features
    stocks_with_returns = calculate_log_returns(stocks_clean)

    # Create cross-sectional features
    stocks_with_market, daily_stats = create_cross_sectional_features(stocks_with_returns, companies)

    # Prepare index data
    index_clean = prepare_index_data(index)

    # Create DistVAE datasets
    datasets = create_distvae_datasets(stocks_with_market, companies)

    # Save datasets
    save_datasets(datasets, index_clean)

    # Generate summary statistics
    generate_summary_statistics(datasets, index_clean)

    print(f"\n=== Data Preparation Complete ===")
    print(f"End time: {datetime.now()}")

    # Print usage instructions
    print("\n=== Usage Instructions ===")
    print("1. Cross-sectional approach (daily market snapshots):")
    print("   - Training data: data/sp500/sp500_cross_sectional_train.csv")
    print("   - Each row = one trading day")
    print("   - Columns = log returns of top 50 stocks + market features")
    print("   - Use for modeling daily market behavior")
    print()
    print("2. Stock-level approach (individual stock characteristics):")
    print("   - Training data: data/sp500/sp500_stock_level_train.csv")
    print("   - Each row = one stock on one day")
    print("   - Features = returns, volatilities, sector dummies")
    print("   - Use for modeling individual stock behavior")
    print()
    print("Next step: Use distvae_standalone.py to train models on these datasets")

if __name__ == "__main__":
    main()