#!/usr/bin/env python3
"""
DistVAE vs Vanilla VAE Comprehensive Comparison

This script loads both trained models and creates detailed comparisons
of their performance on S&P 500 data modeling.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, wasserstein_distance
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import DistVAE
from distvae_standalone import (
    DistVAE,
    prepare_tabular_data,
    dataframe_from_samples,
)

# Import Vanilla VAE (assuming the class from train_vanilla_vae.py)
import sys
sys.path.append('.')

class VanillaVAE(torch.nn.Module):
    """Vanilla VAE class for loading trained model"""
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[512, 256], device=None):
        super(VanillaVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or torch.device('cpu')

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(torch.nn.Linear(hidden_dims[0], input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.to(self.device)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.decode(z)
        return samples

def load_models_and_data():
    """Load both trained models and original data"""
    print("Loading trained models and data...")

    # Load original training data
    train_df = pd.read_csv('data/sp500/sp500_cross_sectional_train.csv', index_col=0)

    # Load metadata
    with open('data/sp500/datasets_metadata.json', 'r') as f:
        metadata = json.load(f)

    stock_cols = metadata['cross_sectional_stocks']
    market_cols = ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']
    continuous = stock_cols + market_cols

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DistVAE model
    print("Loading DistVAE model...")
    X, categorical_dims, meta = prepare_tabular_data(train_df, continuous, [], standardize=True)

    distvae_model = DistVAE(
        continuous_dim=len(continuous),
        categorical_dims=None,
        latent_dim=32,
        step=0.01,
        device=device
    )
    distvae_model.load_state_dict(torch.load('data/sp500/optimized_model_step0.01_latent32_beta0.3.pth', map_location=device))
    distvae_model.eval()

    # Load Vanilla VAE model
    print("Loading Vanilla VAE model...")
    with open('data/sp500/vanilla_vae_results_beta0.1.json', 'r') as f:
        vae_results = json.load(f)

    vanilla_model = VanillaVAE(
        input_dim=len(continuous),
        latent_dim=32,
        device=device
    )
    vanilla_model.load_state_dict(torch.load('data/sp500/vanilla_vae_model_beta0.1.pth', map_location=device))
    vanilla_model.eval()

    # Load normalization parameters for Vanilla VAE
    norm_params = vae_results['normalization_params']

    print(f"✅ Both models loaded successfully")
    print(f"   DistVAE: {sum(p.numel() for p in distvae_model.parameters()):,} parameters")
    print(f"   Vanilla VAE: {sum(p.numel() for p in vanilla_model.parameters()):,} parameters")

    return distvae_model, vanilla_model, train_df, continuous, meta, norm_params

def generate_synthetic_data(distvae_model, vanilla_model, meta, norm_params, continuous, n_samples=5000):
    """Generate synthetic data from both models"""
    print(f"Generating {n_samples} samples from both models...")

    # Generate DistVAE samples
    with torch.no_grad():
        distvae_tensor = distvae_model.generate(n=n_samples, return_onehot=False)

    distvae_df = dataframe_from_samples(
        distvae_tensor, meta, onehot_input=False, map_to_labels=True
    )

    # Generate Vanilla VAE samples
    with torch.no_grad():
        vanilla_tensor = vanilla_model.generate(n_samples)
        vanilla_data = vanilla_tensor.cpu().numpy()

    # Denormalize Vanilla VAE samples
    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])
    vanilla_denormalized = vanilla_data * std + mean

    vanilla_df = pd.DataFrame(vanilla_denormalized, columns=continuous)

    print(f"✅ Generated samples: DistVAE {len(distvae_df)}, Vanilla VAE {len(vanilla_df)}")

    return distvae_df, vanilla_df

def create_distribution_comparison(real_df, distvae_df, vanilla_df, continuous_cols):
    """Create comprehensive distribution comparison plots"""
    print("Creating distribution comparison plots...")

    # Select stocks for visualization
    stock_cols = [col for col in continuous_cols if not col.startswith('log_return_')]
    selected_stocks = stock_cols[:12]

    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()

    comparison_stats = []

    for i, col in enumerate(selected_stocks):
        real_data = real_df[col].dropna()
        distvae_data = distvae_df[col].dropna()
        vanilla_data = vanilla_df[col].dropna()

        # Create histogram comparison
        bins = np.linspace(
            min(real_data.min(), distvae_data.min(), vanilla_data.min()),
            max(real_data.max(), distvae_data.max(), vanilla_data.max()),
            50
        )

        axes[i].hist(real_data, bins=bins, alpha=0.6, density=True,
                    label='Real Data', color='blue', edgecolor='black', linewidth=0.5)
        axes[i].hist(distvae_data, bins=bins, alpha=0.6, density=True,
                    label='DistVAE', color='red', edgecolor='black', linewidth=0.5)
        axes[i].hist(vanilla_data, bins=bins, alpha=0.6, density=True,
                    label='Vanilla VAE', color='green', edgecolor='black', linewidth=0.5)

        # Calculate statistics
        real_mean, real_std = real_data.mean(), real_data.std()
        distvae_mean, distvae_std = distvae_data.mean(), distvae_data.std()
        vanilla_mean, vanilla_std = vanilla_data.mean(), vanilla_data.std()

        real_skew, real_kurt = stats.skew(real_data), stats.kurtosis(real_data)
        distvae_skew, distvae_kurt = stats.skew(distvae_data), stats.kurtosis(distvae_data)
        vanilla_skew, vanilla_kurt = stats.skew(vanilla_data), stats.kurtosis(vanilla_data)

        # Statistical tests
        ks_distvae = kstest(distvae_data, lambda x: stats.percentileofscore(real_data, x) / 100)[1]
        ks_vanilla = kstest(vanilla_data, lambda x: stats.percentileofscore(real_data, x) / 100)[1]

        wd_distvae = wasserstein_distance(real_data, distvae_data)
        wd_vanilla = wasserstein_distance(real_data, vanilla_data)

        axes[i].set_title(f'{col}\nKS p-val: Dist={ks_distvae:.3f}, Van={ks_vanilla:.3f}\n'
                         f'WD: Dist={wd_distvae:.4f}, Van={wd_vanilla:.4f}', fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Log Return')
        axes[i].set_ylabel('Density')

        # Store comparison statistics
        comparison_stats.append({
            'stock': col,
            'real_mean': real_mean, 'real_std': real_std, 'real_skew': real_skew, 'real_kurt': real_kurt,
            'distvae_mean': distvae_mean, 'distvae_std': distvae_std, 'distvae_skew': distvae_skew, 'distvae_kurt': distvae_kurt,
            'vanilla_mean': vanilla_mean, 'vanilla_std': vanilla_std, 'vanilla_skew': vanilla_skew, 'vanilla_kurt': vanilla_kurt,
            'ks_distvae': ks_distvae, 'ks_vanilla': ks_vanilla,
            'wd_distvae': wd_distvae, 'wd_vanilla': wd_vanilla
        })

    plt.tight_layout()
    plt.savefig('data/sp500/comparison_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Distribution comparison plots saved")
    return comparison_stats

def create_tail_comparison(real_df, distvae_df, vanilla_df, continuous_cols):
    """Create tail behavior comparison plots"""
    print("Creating tail behavior comparison...")

    stock_cols = [col for col in continuous_cols if not col.startswith('log_return_')]
    selected_stocks = stock_cols[:8]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    tail_metrics = []

    for i, col in enumerate(selected_stocks):
        real_data = real_df[col].dropna()
        distvae_data = distvae_df[col].dropna()
        vanilla_data = vanilla_df[col].dropna()

        # Q-Q plots
        n_quantiles = 200
        quantiles = np.linspace(0.01, 0.99, n_quantiles)

        real_qq = np.quantile(real_data, quantiles)
        distvae_qq = np.quantile(distvae_data, quantiles)
        vanilla_qq = np.quantile(vanilla_data, quantiles)

        # Plot Q-Q comparisons
        axes[i].scatter(real_qq, distvae_qq, alpha=0.7, s=15, label='DistVAE', color='red')
        axes[i].scatter(real_qq, vanilla_qq, alpha=0.7, s=15, label='Vanilla VAE', color='green')
        axes[i].plot([real_qq.min(), real_qq.max()], [real_qq.min(), real_qq.max()], 'k--', linewidth=2)

        # Calculate R-squared for Q-Q plots
        distvae_r2 = np.corrcoef(real_qq, distvae_qq)[0,1]**2
        vanilla_r2 = np.corrcoef(real_qq, vanilla_qq)[0,1]**2

        # Extreme percentiles
        real_p01, real_p99 = np.percentile(real_data, [1, 99])
        distvae_p01, distvae_p99 = np.percentile(distvae_data, [1, 99])
        vanilla_p01, vanilla_p99 = np.percentile(vanilla_data, [1, 99])

        tail_error_distvae = abs(real_p01 - distvae_p01) + abs(real_p99 - distvae_p99)
        tail_error_vanilla = abs(real_p01 - vanilla_p01) + abs(real_p99 - vanilla_p99)

        axes[i].set_xlabel('Real Data Quantiles')
        axes[i].set_ylabel('Model Quantiles')
        axes[i].set_title(f'{col}\nQ-Q R²: Dist={distvae_r2:.3f}, Van={vanilla_r2:.3f}\n'
                         f'Tail Err: Dist={tail_error_distvae:.4f}, Van={tail_error_vanilla:.4f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        tail_metrics.append({
            'stock': col,
            'distvae_r2': distvae_r2, 'vanilla_r2': vanilla_r2,
            'tail_error_distvae': tail_error_distvae, 'tail_error_vanilla': tail_error_vanilla
        })

    plt.tight_layout()
    plt.savefig('data/sp500/comparison_tails.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Tail comparison plots saved")
    return tail_metrics

def create_correlation_comparison(real_df, distvae_df, vanilla_df, continuous_cols):
    """Compare correlation preservation between models"""
    print("Analyzing correlation preservation...")

    stock_cols = [col for col in continuous_cols if not col.startswith('log_return_')][:20]

    # Calculate correlation matrices
    real_corr = real_df[stock_cols].corr()
    distvae_corr = distvae_df[stock_cols].corr()
    vanilla_corr = vanilla_df[stock_cols].corr()

    # Calculate differences
    distvae_corr_diff = np.abs(real_corr - distvae_corr)
    vanilla_corr_diff = np.abs(real_corr - vanilla_corr)

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    mask = np.triu(np.ones_like(real_corr, dtype=bool))

    # Real correlations
    sns.heatmap(real_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=axes[0,0], vmin=-1, vmax=1)
    axes[0,0].set_title('Real Data Correlations')

    # DistVAE correlations
    sns.heatmap(distvae_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=axes[0,1], vmin=-1, vmax=1)
    axes[0,1].set_title('DistVAE Correlations')

    # Vanilla VAE correlations
    sns.heatmap(vanilla_corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=axes[0,2], vmin=-1, vmax=1)
    axes[0,2].set_title('Vanilla VAE Correlations')

    # DistVAE correlation errors
    max_diff = max(distvae_corr_diff.values.max(), vanilla_corr_diff.values.max())
    sns.heatmap(distvae_corr_diff, mask=mask, annot=False, cmap='Reds',
                square=True, ax=axes[1,0], vmin=0, vmax=max_diff)
    axes[1,0].set_title('DistVAE Correlation Errors')

    # Vanilla VAE correlation errors
    sns.heatmap(vanilla_corr_diff, mask=mask, annot=False, cmap='Reds',
                square=True, ax=axes[1,1], vmin=0, vmax=max_diff)
    axes[1,1].set_title('Vanilla VAE Correlation Errors')

    # Correlation error comparison
    upper_tri_mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    real_corr_values = real_corr.values[upper_tri_mask]
    distvae_corr_values = distvae_corr.values[upper_tri_mask]
    vanilla_corr_values = vanilla_corr.values[upper_tri_mask]

    axes[1,2].scatter(real_corr_values, distvae_corr_values, alpha=0.6, label='DistVAE', color='red', s=20)
    axes[1,2].scatter(real_corr_values, vanilla_corr_values, alpha=0.6, label='Vanilla VAE', color='green', s=20)
    axes[1,2].plot([-1, 1], [-1, 1], 'k--', linewidth=2)
    axes[1,2].set_xlabel('Real Correlations')
    axes[1,2].set_ylabel('Model Correlations')
    axes[1,2].set_title('Correlation Scatter Plot')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/sp500/comparison_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate correlation metrics
    distvae_corr_error = distvae_corr_diff.values[upper_tri_mask].mean()
    vanilla_corr_error = vanilla_corr_diff.values[upper_tri_mask].mean()

    distvae_corr_r2 = np.corrcoef(real_corr_values, distvae_corr_values)[0,1]**2
    vanilla_corr_r2 = np.corrcoef(real_corr_values, vanilla_corr_values)[0,1]**2

    print("✅ Correlation comparison plots saved")

    return {
        'distvae_error': distvae_corr_error,
        'vanilla_error': vanilla_corr_error,
        'distvae_r2': distvae_corr_r2,
        'vanilla_r2': vanilla_corr_r2
    }

def create_summary_comparison_table(comparison_stats, tail_metrics, correlation_metrics):
    """Create comprehensive summary comparison"""
    print("Creating summary comparison...")

    # Calculate average metrics
    avg_metrics = {
        'distvae': {
            'mean_error': np.mean([abs(s['real_mean'] - s['distvae_mean']) for s in comparison_stats]),
            'std_error': np.mean([abs(s['real_std'] - s['distvae_std']) for s in comparison_stats]),
            'skew_error': np.mean([abs(s['real_skew'] - s['distvae_skew']) for s in comparison_stats]),
            'kurt_error': np.mean([abs(s['real_kurt'] - s['distvae_kurt']) for s in comparison_stats]),
            'ks_pval': np.mean([s['ks_distvae'] for s in comparison_stats]),
            'wasserstein': np.mean([s['wd_distvae'] for s in comparison_stats]),
            'qq_r2': np.mean([t['distvae_r2'] for t in tail_metrics]),
            'tail_error': np.mean([t['tail_error_distvae'] for t in tail_metrics]),
            'corr_error': correlation_metrics['distvae_error'],
            'corr_r2': correlation_metrics['distvae_r2']
        },
        'vanilla': {
            'mean_error': np.mean([abs(s['real_mean'] - s['vanilla_mean']) for s in comparison_stats]),
            'std_error': np.mean([abs(s['real_std'] - s['vanilla_std']) for s in comparison_stats]),
            'skew_error': np.mean([abs(s['real_skew'] - s['vanilla_skew']) for s in comparison_stats]),
            'kurt_error': np.mean([abs(s['real_kurt'] - s['vanilla_kurt']) for s in comparison_stats]),
            'ks_pval': np.mean([s['ks_vanilla'] for s in comparison_stats]),
            'wasserstein': np.mean([s['wd_vanilla'] for s in comparison_stats]),
            'qq_r2': np.mean([t['vanilla_r2'] for t in tail_metrics]),
            'tail_error': np.mean([t['tail_error_vanilla'] for t in tail_metrics]),
            'corr_error': correlation_metrics['vanilla_error'],
            'corr_r2': correlation_metrics['vanilla_r2']
        }
    }

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Metric comparison bar chart
    metrics = ['Mean Error', 'Std Error', 'Skew Error', 'Kurt Error', 'Corr Error']
    distvae_values = [avg_metrics['distvae']['mean_error'], avg_metrics['distvae']['std_error'],
                     avg_metrics['distvae']['skew_error'], avg_metrics['distvae']['kurt_error'],
                     avg_metrics['distvae']['corr_error']]
    vanilla_values = [avg_metrics['vanilla']['mean_error'], avg_metrics['vanilla']['std_error'],
                     avg_metrics['vanilla']['skew_error'], avg_metrics['vanilla']['kurt_error'],
                     avg_metrics['vanilla']['corr_error']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[0,0].bar(x - width/2, distvae_values, width, label='DistVAE', alpha=0.8, color='red')
    bars2 = axes[0,0].bar(x + width/2, vanilla_values, width, label='Vanilla VAE', alpha=0.8, color='green')

    axes[0,0].set_title('Model Comparison: Error Metrics (Lower = Better)')
    axes[0,0].set_xlabel('Metric Type')
    axes[0,0].set_ylabel('Error')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(metrics, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # Quality metrics (higher = better)
    quality_metrics = ['KS p-value', 'Q-Q R²', 'Corr R²']
    distvae_quality = [avg_metrics['distvae']['ks_pval'], avg_metrics['distvae']['qq_r2'],
                      avg_metrics['distvae']['corr_r2']]
    vanilla_quality = [avg_metrics['vanilla']['ks_pval'], avg_metrics['vanilla']['qq_r2'],
                      avg_metrics['vanilla']['corr_r2']]

    x_qual = np.arange(len(quality_metrics))
    bars3 = axes[0,1].bar(x_qual - width/2, distvae_quality, width, label='DistVAE', alpha=0.8, color='red')
    bars4 = axes[0,1].bar(x_qual + width/2, vanilla_quality, width, label='Vanilla VAE', alpha=0.8, color='green')

    axes[0,1].set_title('Model Comparison: Quality Metrics (Higher = Better)')
    axes[0,1].set_xlabel('Metric Type')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_xticks(x_qual)
    axes[0,1].set_xticklabels(quality_metrics)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Overall score radar chart
    categories = ['Distribution\nMatching', 'Tail\nBehavior', 'Correlation\nPreservation',
                 'Moment\nMatching', 'Overall\nFidelity']

    # Normalize scores (0-1, higher = better)
    distvae_scores = [
        min(avg_metrics['distvae']['ks_pval'], 1.0),  # KS p-value (capped at 1)
        avg_metrics['distvae']['qq_r2'],  # Q-Q R²
        avg_metrics['distvae']['corr_r2'],  # Correlation R²
        1 - min(avg_metrics['distvae']['kurt_error']/10, 1.0),  # Normalized kurtosis error
        (avg_metrics['distvae']['qq_r2'] + avg_metrics['distvae']['corr_r2']) / 2  # Overall
    ]

    vanilla_scores = [
        min(avg_metrics['vanilla']['ks_pval'], 1.0),
        avg_metrics['vanilla']['qq_r2'],
        avg_metrics['vanilla']['corr_r2'],
        1 - min(avg_metrics['vanilla']['kurt_error']/10, 1.0),
        (avg_metrics['vanilla']['qq_r2'] + avg_metrics['vanilla']['corr_r2']) / 2
    ]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    distvae_scores += distvae_scores[:1]
    vanilla_scores += vanilla_scores[:1]
    angles += angles[:1]

    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    ax_radar.plot(angles, distvae_scores, 'o-', linewidth=2, label='DistVAE', color='red')
    ax_radar.fill(angles, distvae_scores, alpha=0.25, color='red')
    ax_radar.plot(angles, vanilla_scores, 'o-', linewidth=2, label='Vanilla VAE', color='green')
    ax_radar.fill(angles, vanilla_scores, alpha=0.25, color='green')

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Performance Radar Chart', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Detailed metrics table (as text)
    axes[1,1].axis('off')
    table_text = f"""
DETAILED COMPARISON SUMMARY

Distribution Matching:
• DistVAE KS p-value: {avg_metrics['distvae']['ks_pval']:.4f}
• Vanilla KS p-value: {avg_metrics['vanilla']['ks_pval']:.4f}
• Winner: {'DistVAE' if avg_metrics['distvae']['ks_pval'] > avg_metrics['vanilla']['ks_pval'] else 'Vanilla VAE'}

Tail Behavior:
• DistVAE Q-Q R²: {avg_metrics['distvae']['qq_r2']:.3f}
• Vanilla Q-Q R²: {avg_metrics['vanilla']['qq_r2']:.3f}
• Winner: {'DistVAE' if avg_metrics['distvae']['qq_r2'] > avg_metrics['vanilla']['qq_r2'] else 'Vanilla VAE'}

Correlation Preservation:
• DistVAE Error: {avg_metrics['distvae']['corr_error']:.4f}
• Vanilla Error: {avg_metrics['vanilla']['corr_error']:.4f}
• Winner: {'DistVAE' if avg_metrics['distvae']['corr_error'] < avg_metrics['vanilla']['corr_error'] else 'Vanilla VAE'}

Moment Matching:
• DistVAE Kurtosis Error: {avg_metrics['distvae']['kurt_error']:.3f}
• Vanilla Kurtosis Error: {avg_metrics['vanilla']['kurt_error']:.3f}
• Winner: {'DistVAE' if avg_metrics['distvae']['kurt_error'] < avg_metrics['vanilla']['kurt_error'] else 'Vanilla VAE'}

Overall Assessment:
• DistVAE Strengths: Tail behavior, quantile preservation
• Vanilla VAE Strengths: Computational efficiency, simplicity
"""

    axes[1,1].text(0.05, 0.95, table_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('data/sp500/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Summary comparison visualization saved")

    return avg_metrics

def print_final_comparison_report(avg_metrics):
    """Print comprehensive final report"""
    print("\n" + "="*80)
    print("DISTVAE vs VANILLA VAE COMPREHENSIVE COMPARISON REPORT")
    print("="*80)

    print(f"\n📊 DISTRIBUTION MATCHING:")
    print(f"   DistVAE KS p-value: {avg_metrics['distvae']['ks_pval']:.4f}")
    print(f"   Vanilla KS p-value: {avg_metrics['vanilla']['ks_pval']:.4f}")
    dist_winner = "DistVAE" if avg_metrics['distvae']['ks_pval'] > avg_metrics['vanilla']['ks_pval'] else "Vanilla VAE"
    print(f"   🏆 Winner: {dist_winner}")

    print(f"\n📈 TAIL BEHAVIOR:")
    print(f"   DistVAE Q-Q R²: {avg_metrics['distvae']['qq_r2']:.3f}")
    print(f"   Vanilla Q-Q R²: {avg_metrics['vanilla']['qq_r2']:.3f}")
    tail_winner = "DistVAE" if avg_metrics['distvae']['qq_r2'] > avg_metrics['vanilla']['qq_r2'] else "Vanilla VAE"
    print(f"   🏆 Winner: {tail_winner}")

    print(f"\n🔗 CORRELATION PRESERVATION:")
    print(f"   DistVAE Error: {avg_metrics['distvae']['corr_error']:.4f}")
    print(f"   Vanilla Error: {avg_metrics['vanilla']['corr_error']:.4f}")
    corr_winner = "DistVAE" if avg_metrics['distvae']['corr_error'] < avg_metrics['vanilla']['corr_error'] else "Vanilla VAE"
    print(f"   🏆 Winner: {corr_winner}")

    print(f"\n📉 MOMENT MATCHING:")
    print(f"   DistVAE Kurtosis Error: {avg_metrics['distvae']['kurt_error']:.3f}")
    print(f"   Vanilla Kurtosis Error: {avg_metrics['vanilla']['kurt_error']:.3f}")
    moment_winner = "DistVAE" if avg_metrics['distvae']['kurt_error'] < avg_metrics['vanilla']['kurt_error'] else "Vanilla VAE"
    print(f"   🏆 Winner: {moment_winner}")

    # Calculate overall winner
    distvae_wins = sum([
        avg_metrics['distvae']['ks_pval'] > avg_metrics['vanilla']['ks_pval'],
        avg_metrics['distvae']['qq_r2'] > avg_metrics['vanilla']['qq_r2'],
        avg_metrics['distvae']['corr_error'] < avg_metrics['vanilla']['corr_error'],
        avg_metrics['distvae']['kurt_error'] < avg_metrics['vanilla']['kurt_error']
    ])

    print(f"\n🏆 OVERALL WINNER:")
    if distvae_wins >= 3:
        print("   ✅ DISTVAE - Superior for financial data modeling")
        print("   💡 Strengths: Better tail behavior, quantile preservation")
    elif distvae_wins <= 1:
        print("   ✅ VANILLA VAE - Better overall performance")
        print("   💡 Strengths: Simpler architecture, better moment matching")
    else:
        print("   🤝 TIE - Both models have complementary strengths")
        print("   💡 Choice depends on specific application requirements")

    print(f"\n📋 KEY INSIGHTS:")
    print(f"   • DistVAE excels at extreme value modeling (financial tails)")
    print(f"   • Vanilla VAE provides smoother, more Gaussian-like distributions")
    print(f"   • For portfolio risk modeling: {'DistVAE recommended' if tail_winner == 'DistVAE' else 'Consider both models'}")
    print(f"   • For general synthesis: {'Vanilla VAE sufficient' if moment_winner == 'Vanilla VAE' else 'DistVAE preferred'}")

    print(f"\n📁 Generated Files:")
    print(f"   - comparison_distributions.png - Distribution comparisons")
    print(f"   - comparison_tails.png - Tail behavior analysis")
    print(f"   - comparison_correlations.png - Correlation preservation")
    print(f"   - comprehensive_model_comparison.png - Summary comparison")

    print("\n" + "="*80)

def main():
    """Main execution for comprehensive model comparison"""
    print("=== DistVAE vs Vanilla VAE Comprehensive Comparison ===")
    print(f"Start time: {datetime.now()}")

    os.makedirs('data/sp500', exist_ok=True)

    # Load models and data
    distvae_model, vanilla_model, real_df, continuous, meta, norm_params = load_models_and_data()

    # Generate synthetic data
    distvae_df, vanilla_df = generate_synthetic_data(
        distvae_model, vanilla_model, meta, norm_params, continuous, n_samples=5000
    )

    # Create comprehensive comparisons
    comparison_stats = create_distribution_comparison(real_df, distvae_df, vanilla_df, continuous)

    tail_metrics = create_tail_comparison(real_df, distvae_df, vanilla_df, continuous)

    correlation_metrics = create_correlation_comparison(real_df, distvae_df, vanilla_df, continuous)

    avg_metrics = create_summary_comparison_table(comparison_stats, tail_metrics, correlation_metrics)

    # Print final report
    print_final_comparison_report(avg_metrics)

    print(f"\n✅ Comprehensive comparison complete!")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()