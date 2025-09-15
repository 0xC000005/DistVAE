#!/usr/bin/env python3
"""
Optimized S&P 500 DistVAE Training Script

This script implements the recommended improvements for better tail modeling
and correlation preservation:
1. Smaller step sizes (0.01, 0.02) for finer quantile resolution
2. Larger latent dimensions (16, 32) for complex correlation modeling
3. Systematic parameter comparison
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import argparse
from scipy import stats

# Import standalone DistVAE
from distvae_standalone import (
    DistVAE,
    fit_model,
    prepare_tabular_data,
    dataframe_from_samples,
)

def load_sp500_data():
    """Load prepared S&P 500 cross-sectional dataset"""
    print("Loading S&P 500 cross-sectional data...")

    train_df = pd.read_csv('data/sp500/sp500_cross_sectional_train.csv', index_col=0)
    test_df = pd.read_csv('data/sp500/sp500_cross_sectional_test.csv', index_col=0)

    # Load metadata
    with open('data/sp500/datasets_metadata.json', 'r') as f:
        metadata = json.load(f)

    stock_cols = metadata['cross_sectional_stocks']
    market_cols = ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']
    continuous = stock_cols + market_cols

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Total features: {len(continuous)} ({len(stock_cols)} stocks + {len(market_cols)} market)")

    return train_df, test_df, continuous

def prepare_data_for_training(train_df, continuous):
    """Prepare data tensors for DistVAE training"""
    print("Preparing data tensors...")

    X, categorical_dims, meta = prepare_tabular_data(
        train_df, continuous, [], standardize=True
    )

    print(f"Prepared tensor shape: {X.shape}")
    return X, categorical_dims, meta

def train_optimized_model(X_tensor, continuous, step_size=0.01, latent_dim=16, epochs=1500, beta=0.3):
    """Train DistVAE with optimized parameters"""
    print(f"\n{'='*80}")
    print(f"Training DistVAE - Step: {step_size}, Latent: {latent_dim}, Beta: {beta}")
    print(f"{'='*80}")

    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize model with optimized parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DistVAE(
        continuous_dim=len(continuous),
        categorical_dims=None,
        latent_dim=latent_dim,
        step=step_size,
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model with longer training and lower beta for better reconstruction
    print(f"Training for {epochs} epochs...")
    start_time = datetime.now()

    history = fit_model(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=1e-3,
        beta=beta,  # Lower beta emphasizes reconstruction over regularization
        threshold=1e-6,  # Tighter convergence
        device=device,
        verbose=True
    )

    training_time = datetime.now() - start_time
    print(f"Training completed in: {training_time}")

    return model, history

def evaluate_model_quality(model, meta, train_df, continuous, config_name):
    """Quick evaluation of model quality"""
    print(f"Evaluating model quality for {config_name}...")

    # Generate synthetic samples
    n_samples = min(5000, len(train_df))
    with torch.no_grad():
        synthetic_tensor = model.generate(n=n_samples, return_onehot=False)

    synthetic_df = dataframe_from_samples(
        synthetic_tensor, meta, onehot_input=False, map_to_labels=True
    )

    # Quick quality metrics
    stock_cols = [col for col in continuous if not col.startswith('log_return_')][:10]

    metrics = {
        'config': config_name,
        'mean_errors': [],
        'std_errors': [],
        'skew_errors': [],
        'kurt_errors': [],
        'var95_errors': [],
    }

    print(f"\nQuick Quality Assessment - {config_name}:")
    print("-" * 60)
    print(f"{'Stock':<8} {'Mean Err':<10} {'Std Err':<10} {'Skew Err':<10} {'Kurt Err':<10} {'VaR95 Err':<10}")
    print("-" * 60)

    for col in stock_cols:
        real_data = train_df[col].dropna()
        synth_data = synthetic_df[col].dropna()

        # Calculate errors
        mean_err = abs(real_data.mean() - synth_data.mean())
        std_err = abs(real_data.std() - synth_data.std())
        skew_err = abs(stats.skew(real_data) - stats.skew(synth_data))
        kurt_err = abs(stats.kurtosis(real_data) - stats.kurtosis(synth_data))

        real_var95 = np.percentile(real_data, 5)
        synth_var95 = np.percentile(synth_data, 5)
        var95_err = abs(real_var95 - synth_var95)

        metrics['mean_errors'].append(mean_err)
        metrics['std_errors'].append(std_err)
        metrics['skew_errors'].append(skew_err)
        metrics['kurt_errors'].append(kurt_err)
        metrics['var95_errors'].append(var95_err)

        print(f"{col:<8} {mean_err:<10.4f} {std_err:<10.4f} {skew_err:<10.4f} {kurt_err:<10.4f} {var95_err:<10.4f}")

    # Summary statistics
    avg_metrics = {
        'avg_mean_error': np.mean(metrics['mean_errors']),
        'avg_std_error': np.mean(metrics['std_errors']),
        'avg_skew_error': np.mean(metrics['skew_errors']),
        'avg_kurt_error': np.mean(metrics['kurt_errors']),
        'avg_var95_error': np.mean(metrics['var95_errors']),
    }

    print("\nSummary:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    # Correlation analysis
    stock_subset = stock_cols
    real_corr = train_df[stock_subset].corr()
    synth_corr = synthetic_df[stock_subset].corr()
    corr_diff = np.abs(real_corr - synth_corr)

    upper_tri_mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    avg_corr_error = corr_diff.values[upper_tri_mask].mean()

    print(f"Average correlation error: {avg_corr_error:.4f}")

    metrics.update(avg_metrics)
    metrics['avg_corr_error'] = avg_corr_error

    return metrics, synthetic_df

def save_model_results(model, history, synthetic_df, meta, config_name):
    """Save trained model and results"""
    print(f"Saving results for {config_name}...")

    # Save model
    model_path = f'data/sp500/optimized_model_{config_name}.pth'
    torch.save(model.state_dict(), model_path)

    # Save synthetic data
    synthetic_df.to_csv(f'data/sp500/optimized_synthetic_{config_name}.csv', index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'data/sp500/optimized_history_{config_name}.csv', index=False)

    print(f"Results saved with prefix: optimized_{config_name}")

def create_comparison_plots(all_metrics, all_histories):
    """Create comparison plots for different configurations"""
    print("Creating comparison plots...")

    # 1. Training loss comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss comparison
    for config, history in all_histories.items():
        epochs = range(1, len(history) + 1)
        losses = [h['loss'] for h in history]
        axes[0,0].plot(epochs, losses, label=config, alpha=0.8)

    axes[0,0].set_title('Training Loss Comparison')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Total Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Quality metrics comparison
    configs = list(all_metrics.keys())
    mean_errors = [all_metrics[c]['avg_mean_error'] for c in configs]
    kurt_errors = [all_metrics[c]['avg_kurt_error'] for c in configs]
    corr_errors = [all_metrics[c]['avg_corr_error'] for c in configs]
    var95_errors = [all_metrics[c]['avg_var95_error'] for c in configs]

    x = np.arange(len(configs))
    width = 0.15

    axes[0,1].bar(x - 1.5*width, mean_errors, width, label='Mean Error', alpha=0.8)
    axes[0,1].bar(x - 0.5*width, corr_errors, width, label='Correlation Error', alpha=0.8)
    axes[0,1].bar(x + 0.5*width, var95_errors, width, label='VaR95 Error', alpha=0.8)

    axes[0,1].set_title('Quality Metrics Comparison')
    axes[0,1].set_xlabel('Configuration')
    axes[0,1].set_ylabel('Error')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(configs, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Kurtosis error (tail behavior)
    axes[1,0].bar(configs, kurt_errors, alpha=0.8, color='red')
    axes[1,0].set_title('Kurtosis Error (Tail Behavior)')
    axes[1,0].set_xlabel('Configuration')
    axes[1,0].set_ylabel('Kurtosis Error')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)

    # Overall score (lower is better)
    overall_scores = []
    for config in configs:
        m = all_metrics[config]
        # Weighted combination of key metrics
        score = (m['avg_kurt_error'] * 0.4 +  # Tail behavior (most important)
                m['avg_corr_error'] * 10 +    # Correlation preservation
                m['avg_var95_error'] * 100 +  # Extreme value accuracy
                m['avg_mean_error'] * 100)    # Basic moment matching
        overall_scores.append(score)

    best_idx = np.argmin(overall_scores)
    colors = ['green' if i == best_idx else 'blue' for i in range(len(configs))]

    axes[1,1].bar(configs, overall_scores, alpha=0.8, color=colors)
    axes[1,1].set_title('Overall Quality Score (Lower = Better)')
    axes[1,1].set_xlabel('Configuration')
    axes[1,1].set_ylabel('Composite Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)

    # Highlight best configuration
    axes[1,1].text(best_idx, overall_scores[best_idx] + max(overall_scores)*0.05,
                   '🏆 BEST', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('data/sp500/optimized_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Best configuration: {configs[best_idx]} (Score: {overall_scores[best_idx]:.4f})")
    return configs[best_idx], overall_scores[best_idx]

def main():
    """Main training execution with parameter optimization"""
    print("=== Optimized S&P 500 DistVAE Training ===")
    print(f"Start time: {datetime.now()}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized DistVAE models')
    parser.add_argument('--quick', action='store_true', help='Quick training with fewer epochs')
    parser.add_argument('--step_sizes', nargs='+', type=float, default=[0.01, 0.02],
                       help='Step sizes to test')
    parser.add_argument('--latent_dims', nargs='+', type=int, default=[16, 32],
                       help='Latent dimensions to test')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('--beta', type=float, default=0.3, help='KL weight parameter')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 200
        print("🚀 Quick mode: Using 200 epochs for faster testing")

    # Ensure output directory
    os.makedirs('data/sp500', exist_ok=True)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    train_df, test_df, continuous = load_sp500_data()
    X_tensor, categorical_dims, meta = prepare_data_for_training(train_df, continuous)

    # Test different configurations
    configurations = []
    for step_size in args.step_sizes:
        for latent_dim in args.latent_dims:
            config_name = f"step{step_size:.2f}_latent{latent_dim}_beta{args.beta:.1f}"
            configurations.append((step_size, latent_dim, config_name))

    print(f"\n🧪 Testing {len(configurations)} configurations:")
    for step, latent, name in configurations:
        print(f"   - {name}")

    all_metrics = {}
    all_histories = {}

    # Train each configuration
    for i, (step_size, latent_dim, config_name) in enumerate(configurations):
        print(f"\n🔄 Configuration {i+1}/{len(configurations)}: {config_name}")

        try:
            # Train model
            model, history = train_optimized_model(
                X_tensor, continuous,
                step_size=step_size,
                latent_dim=latent_dim,
                epochs=args.epochs,
                beta=args.beta
            )

            # Evaluate quality
            metrics, synthetic_df = evaluate_model_quality(
                model, meta, train_df, continuous, config_name
            )

            # Save results
            save_model_results(model, history, synthetic_df, meta, config_name)

            all_metrics[config_name] = metrics
            all_histories[config_name] = history

            print(f"✅ {config_name} completed successfully")

        except Exception as e:
            print(f"❌ {config_name} failed: {str(e)}")
            continue

    # Create comparison analysis
    if len(all_metrics) > 1:
        print(f"\n📊 Creating comparison analysis...")
        best_config, best_score = create_comparison_plots(all_metrics, all_histories)

        # Final recommendations
        print(f"\n🏆 RESULTS SUMMARY:")
        print(f"{'='*60}")
        print(f"Best Configuration: {best_config}")
        print(f"Quality Score: {best_score:.4f}")

        print(f"\n📈 Quality Comparison:")
        for config, metrics in all_metrics.items():
            marker = "🏆" if config == best_config else "  "
            print(f"{marker} {config}:")
            print(f"     Kurtosis Error: {metrics['avg_kurt_error']:.3f}")
            print(f"     Correlation Error: {metrics['avg_corr_error']:.4f}")
            print(f"     VaR95 Error: {metrics['avg_var95_error']:.4f}")

        print(f"\n💡 RECOMMENDATIONS:")
        best_metrics = all_metrics[best_config]
        if best_metrics['avg_kurt_error'] > 2.0:
            print("   🔧 Still high kurtosis error - consider step size < 0.01")
        if best_metrics['avg_corr_error'] > 0.05:
            print("   🔧 Correlation preservation could improve - try larger latent_dim")
        if best_metrics['avg_var95_error'] > 0.01:
            print("   🔧 Tail modeling needs work - try longer training or β < 0.2")

        print(f"\n📁 Files generated:")
        print(f"   - Models: optimized_model_*.pth")
        print(f"   - Synthetic data: optimized_synthetic_*.csv")
        print(f"   - Training history: optimized_history_*.csv")
        print(f"   - Comparison plots: optimized_model_comparison.png")

    else:
        print(f"⚠️  Only one configuration completed - no comparison available")

    print(f"\n✅ Optimization complete!")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()