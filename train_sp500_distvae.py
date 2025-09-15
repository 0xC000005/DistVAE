#!/usr/bin/env python3
"""
Train DistVAE on S&P 500 Log Returns Data

This script demonstrates how to train the standalone DistVAE on prepared S&P 500 data
for modeling equity log returns distributions.
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

# Import standalone DistVAE
from distvae_standalone import (
    DistVAE,
    fit_model,
    prepare_tabular_data,
    dataframe_from_samples,
)

def load_sp500_data():
    """
    Load prepared S&P 500 cross-sectional dataset for modeling stock correlations
    """
    print("Loading S&P 500 cross-sectional data...")

    train_df = pd.read_csv('data/sp500/sp500_cross_sectional_train.csv', index_col=0)
    test_df = pd.read_csv('data/sp500/sp500_cross_sectional_test.csv', index_col=0)

    # Separate continuous (stock returns) and market features
    with open('data/sp500/datasets_metadata.json', 'r') as f:
        metadata = json.load(f)

    stock_cols = metadata['cross_sectional_stocks']
    market_cols = ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']

    # All features are continuous for cross-sectional approach
    continuous = stock_cols + market_cols
    discrete = []

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Continuous features ({len(continuous)}): {continuous[:5]}{'...' if len(continuous) > 5 else ''}")
    print(f"Discrete features ({len(discrete)}): {discrete[:5]}{'...' if len(discrete) > 5 else ''}")

    return train_df, test_df, continuous, discrete

def prepare_data_for_distvae(train_df, continuous, discrete):
    """Prepare data tensors for DistVAE training"""
    print("Preparing data tensors for DistVAE...")

    # Cross-sectional approach: all columns are continuous (stock returns + market features)
    X, categorical_dims, meta = prepare_tabular_data(
        train_df, continuous, [], standardize=True
    )

    print(f"Prepared tensor shape: {X.shape}")
    print(f"Categorical dimensions: {categorical_dims}")

    return X, categorical_dims, meta

def train_distvae_on_sp500(epochs=50, beta=0.5):
    """Train DistVAE on S&P 500 cross-sectional data for modeling stock correlations"""
    print("=== Training DistVAE on S&P 500 (Cross-Sectional) ===")

    # Load data
    train_df, test_df, continuous, discrete = load_sp500_data()

    # Prepare tensors
    X_tensor, categorical_dims, meta = prepare_data_for_distvae(train_df, continuous, discrete)

    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DistVAE(
        continuous_dim=len(continuous),
        categorical_dims=categorical_dims,
        latent_dim=8,  # Higher dimensional latent space for financial data
        step=0.05,     # Finer quantile discretization
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = fit_model(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=1e-3,
        beta=beta,
        threshold=1e-5,
        device=device,
        verbose=True
    )

    # Generate synthetic samples
    print("\nGenerating synthetic samples...")
    n_samples = min(10000, len(train_df))  # Generate up to 10k samples

    with torch.no_grad():
        synthetic_tensor = model.generate(n=n_samples, return_onehot=(len(categorical_dims) > 0))

    # Convert back to DataFrame
    synthetic_df = dataframe_from_samples(
        synthetic_tensor, meta,
        onehot_input=(len(categorical_dims) > 0),
        map_to_labels=True
    )

    print(f"Generated {len(synthetic_df)} synthetic samples")

    return model, history, synthetic_df, train_df, test_df, meta

def evaluate_synthetic_quality(synthetic_df, real_df, continuous_cols):
    """Evaluate quality of synthetic samples"""
    print("\n=== Evaluating Synthetic Data Quality ===")

    # Statistical comparison
    real_stats = real_df[continuous_cols].describe()
    synth_stats = synthetic_df[continuous_cols].describe()

    print("\nMoment Matching (first 5 features):")
    comparison_cols = continuous_cols[:5]
    for col in comparison_cols:
        real_mean, real_std = real_stats.loc['mean', col], real_stats.loc['std', col]
        synth_mean, synth_std = synth_stats.loc['mean', col], synth_stats.loc['std', col]

        print(f"{col:20s}: Real μ={real_mean:7.4f} σ={real_std:7.4f} | "
              f"Synth μ={synth_mean:7.4f} σ={synth_std:7.4f}")

    # Distribution comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    plot_cols = continuous_cols[:6]  # Plot first 6 features

    for i, col in enumerate(plot_cols):
        if col in real_df.columns and col in synthetic_df.columns:
            # Histogram comparison
            axes[i].hist(real_df[col].dropna(), bins=50, alpha=0.7,
                        density=True, label='Real', color='blue')
            axes[i].hist(synthetic_df[col].dropna(), bins=50, alpha=0.7,
                        density=True, label='Synthetic', color='red')
            axes[i].set_title(f'{col}')
            axes[i].legend()
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')

    plt.tight_layout()
    save_path = 'data/sp500/distvae_evaluation_cross_sectional.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution comparison plots saved to: {save_path}")

    # Correlation analysis
    if len(continuous_cols) > 1:
        print("\nCorrelation Analysis:")

        # Select subset of stocks for correlation analysis
        stock_subset = continuous_cols[:10]  # First 10 stocks

        real_corr = real_df[stock_subset].corr()
        synth_corr = synthetic_df[stock_subset].corr()

        # Correlation difference
        corr_diff = np.abs(real_corr - synth_corr)
        print(f"Mean absolute correlation error: {corr_diff.values[np.triu_indices_from(corr_diff, k=1)].mean():.4f}")

        # Plot correlation matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(real_corr, annot=False, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Real Data Correlations')

        sns.heatmap(synth_corr, annot=False, cmap='coolwarm', center=0, ax=axes[1])
        axes[1].set_title('Synthetic Data Correlations')

        sns.heatmap(corr_diff, annot=False, cmap='Reds', ax=axes[2])
        axes[2].set_title('Absolute Correlation Differences')

        plt.tight_layout()
        plt.savefig('data/sp500/correlation_analysis_cross_sectional.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Correlation analysis saved to: data/sp500/correlation_analysis_cross_sectional.png")

def save_model_and_results(model, history, synthetic_df, meta):
    """Save trained model and results"""
    print("\nSaving model...")

    # Save model
    model_path = 'data/sp500/distvae_model_cross_sectional.pth'
    torch.save(model.state_dict(), model_path)

    # Save synthetic data
    synthetic_df.to_csv('data/sp500/synthetic_data_cross_sectional.csv', index=False)

    print(f"Model saved to: {model_path}")
    print("Synthetic data saved to: data/sp500/synthetic_data_cross_sectional.csv")

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history) + 1)

    # Loss
    losses = [h['loss'] for h in history]
    axes[0].plot(epochs, losses)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    # Quantile loss
    quantile_losses = [h['quantile'] for h in history]
    axes[1].plot(epochs, quantile_losses)
    axes[1].set_title('Quantile Loss (CRPS)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    # KL divergence
    kl_losses = [h['KL'] for h in history]
    axes[2].plot(epochs, kl_losses)
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')

    plt.tight_layout()
    plt.savefig('data/sp500/training_history_cross_sectional.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Training history plot saved to: data/sp500/training_history_cross_sectional.png")

def main():
    """Main execution function"""
    print("=== S&P 500 DistVAE Training ===")
    print(f"Start time: {datetime.now()}")

    # Ensure output directory exists
    os.makedirs('data/sp500', exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Train cross-sectional approach for stock correlation modeling
    print(f"\n{'='*60}")
    print("Training Cross-Sectional DistVAE for Stock Correlations")
    print(f"{'='*60}")

    try:
        # Train model
        model, history, synthetic_df, train_df, test_df, meta = train_distvae_on_sp500(
            epochs=1000, beta=0.5
        )

        # Get continuous columns for evaluation
        with open('data/sp500/datasets_metadata.json', 'r') as f:
            metadata = json.load(f)
        continuous = metadata['cross_sectional_stocks'] + [
            'log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean'
        ]

        # Evaluate synthetic data quality
        evaluate_synthetic_quality(synthetic_df, train_df, continuous)

        # Plot training history
        plot_training_history(history)

        # Save model and results
        save_model_and_results(model, history, synthetic_df, meta)

        print("\nCompleted cross-sectional training")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    print(f"\n=== Training Complete ===")
    print(f"End time: {datetime.now()}")
    print("\nResults saved in data/sp500/:")
    print("- Model: distvae_model_cross_sectional.pth")
    print("- Synthetic data: synthetic_data_cross_sectional.csv")
    print("- Evaluation plots: distvae_evaluation_cross_sectional.png")
    print("- Training history: training_history_cross_sectional.png")
    print("- Correlation analysis: correlation_analysis_cross_sectional.png")

if __name__ == "__main__":
    main()