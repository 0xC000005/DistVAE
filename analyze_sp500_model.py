#!/usr/bin/env python3
"""
S&P 500 DistVAE Model Analysis Script

This script loads the trained DistVAE model and performs comprehensive analysis
including distribution comparisons, correlation analysis, tail behavior assessment,
and statistical quality tests.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson, jarque_bera, shapiro
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import standalone DistVAE
from distvae_standalone import (
    DistVAE,
    prepare_tabular_data,
    dataframe_from_samples,
)

def load_trained_model():
    """Load the trained DistVAE model and metadata"""
    print("Loading trained DistVAE model...")

    # Load model metadata
    with open('data/sp500/datasets_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load training data to get model dimensions
    train_df = pd.read_csv('data/sp500/sp500_cross_sectional_train.csv', index_col=0)

    stock_cols = metadata['cross_sectional_stocks']
    market_cols = ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']
    continuous = stock_cols + market_cols

    # Prepare data to get meta info
    X, categorical_dims, meta = prepare_tabular_data(
        train_df, continuous, [], standardize=True
    )

    # Initialize model with same architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DistVAE(
        continuous_dim=len(continuous),
        categorical_dims=None,
        latent_dim=8,
        step=0.05,
        device=device
    )

    # Load trained weights
    model_path = 'data/sp500/distvae_model_cross_sectional.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded successfully from: {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, meta, continuous, train_df

def generate_synthetic_data(model, meta, n_samples=10000):
    """Generate synthetic samples from trained model"""
    print(f"Generating {n_samples} synthetic samples...")

    with torch.no_grad():
        synthetic_tensor = model.generate(n=n_samples, return_onehot=False)

    # Convert back to DataFrame
    synthetic_df = dataframe_from_samples(
        synthetic_tensor, meta,
        onehot_input=False,
        map_to_labels=True
    )

    print(f"Generated {len(synthetic_df)} synthetic samples")
    return synthetic_df

def create_distribution_comparison(real_df, synthetic_df, continuous_cols, n_plots=12):
    """Create comprehensive distribution comparison plots"""
    print("Creating distribution comparison plots...")

    # Select subset of stocks for visualization
    plot_cols = continuous_cols[:n_plots]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i, col in enumerate(plot_cols):
        if i >= len(axes):
            break

        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()

        # Histogram comparison
        axes[i].hist(real_data, bins=50, alpha=0.7, density=True,
                    label='Real', color='blue', edgecolor='black', linewidth=0.5)
        axes[i].hist(synth_data, bins=50, alpha=0.7, density=True,
                    label='Synthetic', color='red', edgecolor='black', linewidth=0.5)

        # Add statistics
        real_mean, real_std = real_data.mean(), real_data.std()
        synth_mean, synth_std = synth_data.mean(), synth_data.std()

        axes[i].set_title(f'{col}\nReal: μ={real_mean:.4f}, σ={real_std:.4f}\n'
                         f'Synth: μ={synth_mean:.4f}, σ={synth_std:.4f}', fontsize=10)
        axes[i].legend()
        axes[i].set_xlabel('Log Return')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'data/sp500/analysis_distribution_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution comparison saved to: {save_path}")
    return save_path

def analyze_correlations(real_df, synthetic_df, continuous_cols, n_stocks=20):
    """Analyze and compare correlation matrices"""
    print("Analyzing correlation matrices...")

    # Select subset for correlation analysis
    stock_subset = [col for col in continuous_cols if not col.startswith('log_return_')][:n_stocks]

    real_corr = real_df[stock_subset].corr()
    synth_corr = synthetic_df[stock_subset].corr()

    # Calculate correlation differences
    corr_diff = np.abs(real_corr - synth_corr)

    # Correlation statistics
    upper_tri_mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    real_corr_values = real_corr.values[upper_tri_mask]
    synth_corr_values = synth_corr.values[upper_tri_mask]
    corr_diff_values = corr_diff.values[upper_tri_mask]

    print(f"Correlation Analysis Summary:")
    print(f"Mean absolute correlation error: {corr_diff_values.mean():.4f}")
    print(f"Max correlation error: {corr_diff_values.max():.4f}")
    print(f"Correlation of correlations: {np.corrcoef(real_corr_values, synth_corr_values)[0,1]:.4f}")

    # Create correlation plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Real correlations
    mask = np.triu(np.ones_like(real_corr, dtype=bool))
    sns.heatmap(real_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=axes[0,0], vmin=-1, vmax=1)
    axes[0,0].set_title('Real Data Correlations')

    # Synthetic correlations
    sns.heatmap(synth_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=axes[0,1], vmin=-1, vmax=1)
    axes[0,1].set_title('Synthetic Data Correlations')

    # Correlation differences
    sns.heatmap(corr_diff, mask=mask, annot=False, cmap='Reds',
                square=True, ax=axes[1,0], vmin=0, vmax=corr_diff_values.max())
    axes[1,0].set_title('Absolute Correlation Differences')

    # Scatter plot of correlations
    axes[1,1].scatter(real_corr_values, synth_corr_values, alpha=0.6)
    axes[1,1].plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    axes[1,1].set_xlabel('Real Correlations')
    axes[1,1].set_ylabel('Synthetic Correlations')
    axes[1,1].set_title(f'Correlation Scatter (r={np.corrcoef(real_corr_values, synth_corr_values)[0,1]:.3f})')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'data/sp500/analysis_correlation_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Correlation analysis saved to: {save_path}")
    return corr_diff_values.mean(), corr_diff_values.max()

def analyze_tail_behavior(real_df, synthetic_df, continuous_cols, n_stocks=8):
    """Analyze tail behavior and extreme value statistics"""
    print("Analyzing tail behavior...")

    # Select stocks for tail analysis
    stock_cols = [col for col in continuous_cols if not col.startswith('log_return_')][:n_stocks]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    tail_stats = []

    for i, col in enumerate(stock_cols):
        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()

        # Q-Q plot
        real_sorted = np.sort(real_data)
        synth_sorted = np.sort(synth_data)

        # Match lengths for Q-Q plot
        min_len = min(len(real_sorted), len(synth_sorted))
        real_quantiles = np.linspace(0, 1, min_len)
        real_qq = np.quantile(real_sorted, real_quantiles)
        synth_qq = np.quantile(synth_sorted, real_quantiles)

        axes[i].scatter(real_qq, synth_qq, alpha=0.6, s=1)
        axes[i].plot([real_qq.min(), real_qq.max()], [real_qq.min(), real_qq.max()], 'r--', linewidth=2)
        axes[i].set_xlabel('Real Data Quantiles')
        axes[i].set_ylabel('Synthetic Data Quantiles')
        axes[i].set_title(f'{col} Q-Q Plot')
        axes[i].grid(True, alpha=0.3)

        # Tail statistics
        real_p99 = np.percentile(real_data, 99)
        real_p01 = np.percentile(real_data, 1)
        synth_p99 = np.percentile(synth_data, 99)
        synth_p01 = np.percentile(synth_data, 1)

        # Extreme value analysis
        real_skew = stats.skew(real_data)
        real_kurt = stats.kurtosis(real_data)
        synth_skew = stats.skew(synth_data)
        synth_kurt = stats.kurtosis(synth_data)

        tail_stats.append({
            'stock': col,
            'real_p01': real_p01,
            'real_p99': real_p99,
            'synth_p01': synth_p01,
            'synth_p99': synth_p99,
            'real_skew': real_skew,
            'real_kurt': real_kurt,
            'synth_skew': synth_skew,
            'synth_kurt': synth_kurt
        })

    plt.tight_layout()
    save_path = 'data/sp500/analysis_tail_behavior.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print tail statistics
    print("\nTail Behavior Analysis:")
    print("=" * 80)
    print(f"{'Stock':<8} {'Real P1%':<10} {'Synth P1%':<10} {'Real P99%':<10} {'Synth P99%':<10} {'Skew Diff':<10} {'Kurt Diff':<10}")
    print("-" * 80)

    for stat in tail_stats:
        skew_diff = abs(stat['real_skew'] - stat['synth_skew'])
        kurt_diff = abs(stat['real_kurt'] - stat['synth_kurt'])
        print(f"{stat['stock']:<8} {stat['real_p01']:<10.4f} {stat['synth_p01']:<10.4f} "
              f"{stat['real_p99']:<10.4f} {stat['synth_p99']:<10.4f} {skew_diff:<10.4f} {kurt_diff:<10.4f}")

    print(f"\nTail behavior analysis saved to: {save_path}")
    return tail_stats

def statistical_quality_tests(real_df, synthetic_df, continuous_cols, n_stocks=10):
    """Perform comprehensive statistical quality tests"""
    print("Performing statistical quality tests...")

    # Select stocks for testing
    test_stocks = [col for col in continuous_cols if not col.startswith('log_return_')][:n_stocks]

    test_results = []

    print("\nStatistical Quality Tests:")
    print("=" * 100)
    print(f"{'Stock':<8} {'KS p-val':<10} {'AD Stat':<10} {'JB p-val':<10} {'SW p-val':<10} {'Mean Diff':<10} {'Std Diff':<10}")
    print("-" * 100)

    for col in test_stocks:
        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = kstest(synth_data, lambda x: stats.percentileofscore(real_data, x) / 100)

        # Anderson-Darling test (compare to normal distribution)
        ad_real = anderson(real_data, dist='norm')
        ad_synth = anderson(synth_data, dist='norm')

        # Jarque-Bera test for normality
        jb_real_stat, jb_real_pval = jarque_bera(real_data)
        jb_synth_stat, jb_synth_pval = jarque_bera(synth_data)

        # Shapiro-Wilk test (if sample size allows)
        if len(real_data) <= 5000 and len(synth_data) <= 5000:
            sw_real_stat, sw_real_pval = shapiro(real_data[:5000])
            sw_synth_stat, sw_synth_pval = shapiro(synth_data[:5000])
        else:
            sw_real_pval = sw_synth_pval = np.nan

        # Moment differences
        mean_diff = abs(real_data.mean() - synth_data.mean())
        std_diff = abs(real_data.std() - synth_data.std())

        test_results.append({
            'stock': col,
            'ks_pval': ks_pval,
            'ad_real': ad_real.statistic,
            'ad_synth': ad_synth.statistic,
            'jb_real_pval': jb_real_pval,
            'jb_synth_pval': jb_synth_pval,
            'sw_real_pval': sw_real_pval,
            'sw_synth_pval': sw_synth_pval,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        })

        print(f"{col:<8} {ks_pval:<10.4f} {ad_synth.statistic:<10.2f} {jb_synth_pval:<10.4f} "
              f"{sw_synth_pval:<10.4f} {mean_diff:<10.4f} {std_diff:<10.4f}")

    return test_results

def create_summary_report(corr_mae, corr_max, tail_stats, test_results):
    """Create comprehensive summary report"""
    print("\n" + "="*80)
    print("DISTVAE MODEL QUALITY ASSESSMENT SUMMARY")
    print("="*80)

    print(f"\n📊 CORRELATION ANALYSIS:")
    print(f"   Mean Absolute Error: {corr_mae:.4f}")
    print(f"   Maximum Error: {corr_max:.4f}")
    print(f"   Quality: {'✅ Excellent' if corr_mae < 0.05 else '⚠️  Good' if corr_mae < 0.1 else '❌ Poor'}")

    print(f"\n📈 TAIL BEHAVIOR:")
    avg_skew_diff = np.mean([abs(s['real_skew'] - s['synth_skew']) for s in tail_stats])
    avg_kurt_diff = np.mean([abs(s['real_kurt'] - s['synth_kurt']) for s in tail_stats])
    print(f"   Average Skewness Difference: {avg_skew_diff:.4f}")
    print(f"   Average Kurtosis Difference: {avg_kurt_diff:.4f}")
    print(f"   Fat-tail Preservation: {'✅ Good' if avg_kurt_diff < 1.0 else '⚠️  Moderate' if avg_kurt_diff < 2.0 else '❌ Poor'}")

    print(f"\n🔬 STATISTICAL TESTS:")
    avg_ks_pval = np.mean([r['ks_pval'] for r in test_results])
    avg_mean_diff = np.mean([r['mean_diff'] for r in test_results])
    avg_std_diff = np.mean([r['std_diff'] for r in test_results])

    print(f"   Average KS Test p-value: {avg_ks_pval:.4f}")
    print(f"   Average Mean Difference: {avg_mean_diff:.6f}")
    print(f"   Average Std Difference: {avg_std_diff:.6f}")
    print(f"   Distribution Matching: {'✅ Excellent' if avg_ks_pval > 0.05 else '⚠️  Acceptable' if avg_ks_pval > 0.01 else '❌ Poor'}")

    print(f"\n🏆 OVERALL ASSESSMENT:")
    scores = []
    scores.append(1 if corr_mae < 0.05 else 0.5 if corr_mae < 0.1 else 0)
    scores.append(1 if avg_kurt_diff < 1.0 else 0.5 if avg_kurt_diff < 2.0 else 0)
    scores.append(1 if avg_ks_pval > 0.05 else 0.5 if avg_ks_pval > 0.01 else 0)

    overall_score = np.mean(scores)
    if overall_score > 0.8:
        assessment = "✅ EXCELLENT - Model produces high-quality synthetic data"
    elif overall_score > 0.6:
        assessment = "⚠️  GOOD - Model performance is acceptable with minor issues"
    else:
        assessment = "❌ NEEDS IMPROVEMENT - Model requires further tuning"

    print(f"   {assessment}")
    print(f"   Quality Score: {overall_score:.2f}/1.00")

    print("\n" + "="*80)

def main():
    """Main analysis execution"""
    print("=== S&P 500 DistVAE Model Analysis ===")
    print(f"Start time: {datetime.now()}")

    # Ensure output directory
    os.makedirs('data/sp500', exist_ok=True)

    # Load model and data
    model, meta, continuous_cols, real_df = load_trained_model()

    # Generate synthetic data
    synthetic_df = generate_synthetic_data(model, meta, n_samples=10000)

    # Perform analyses
    create_distribution_comparison(real_df, synthetic_df, continuous_cols)

    corr_mae, corr_max = analyze_correlations(real_df, synthetic_df, continuous_cols)

    tail_stats = analyze_tail_behavior(real_df, synthetic_df, continuous_cols)

    test_results = statistical_quality_tests(real_df, synthetic_df, continuous_cols)

    # Create summary report
    create_summary_report(corr_mae, corr_max, tail_stats, test_results)

    print(f"\n✅ Analysis complete! Results saved in data/sp500/analysis_*.png")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()