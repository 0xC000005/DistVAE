#!/usr/bin/env python3
"""
Interest Rate DistVAE Training Script

This script trains a DistVAE model on FRB H15 Treasury yield curve data
to predict interest rates across different maturities.

The model learns the distributional relationships between different maturities
and can generate synthetic yield curves or predict missing rates.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import standalone DistVAE
from distvae_standalone import (
    DistVAE,
    fit_model,
    prepare_tabular_data,
    dataframe_from_samples,
)

def load_and_clean_ir_data():
    """Load and clean FRB H15 interest rate data"""
    print("Loading FRB H15 interest rate data...")

    # Load the data, skipping metadata rows
    df = pd.read_csv('data/ir/FRB_H15.csv', skiprows=5)

    # Rename columns for easier handling
    maturity_mapping = {
        'RIFLGFCM01_N.B': '1M',
        'RIFLGFCM03_N.B': '3M',
        'RIFLGFCM06_N.B': '6M',
        'RIFLGFCY01_N.B': '1Y',
        'RIFLGFCY02_N.B': '2Y',
        'RIFLGFCY03_N.B': '3Y',
        'RIFLGFCY05_N.B': '5Y',
        'RIFLGFCY07_N.B': '7Y',
        'RIFLGFCY10_N.B': '10Y',
        'RIFLGFCY20_N.B': '20Y',
        'RIFLGFCY30_N.B': '30Y'
    }

    df = df.rename(columns=maturity_mapping)
    df = df.rename(columns={'Time Period': 'Date'})

    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Replace 'ND' with NaN and convert to numeric
    maturity_cols = list(maturity_mapping.values())
    for col in maturity_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Check missing data before transformation
    missing_pct = df[maturity_cols].isnull().mean() * 100
    print(f"\nMissing data percentage by maturity (levels):")
    for col, pct in missing_pct.items():
        print(f"  {col}: {pct:.1f}%")

    return df, maturity_cols

def convert_to_daily_changes(df, maturity_cols):
    """Convert yield levels to daily changes"""
    print("Converting yield levels to daily changes...")

    df_changes = df.copy()

    # Calculate daily changes for each maturity
    for col in maturity_cols:
        df_changes[f'{col}_change'] = df_changes[col].diff()

    # Create new maturity columns list with _change suffix
    change_cols = [f'{col}_change' for col in maturity_cols]

    # Drop the first row (NaN due to diff) and original level columns
    df_changes = df_changes.dropna(subset=change_cols, how='all').copy()
    df_changes = df_changes.drop(columns=maturity_cols)

    print(f"Shape after converting to daily changes: {df_changes.shape}")
    print(f"Date range after dropping first row: {df_changes['Date'].min()} to {df_changes['Date'].max()}")

    # Check missing data after transformation
    missing_pct = df_changes[change_cols].isnull().mean() * 100
    print(f"\nMissing data percentage by maturity (changes):")
    for col, pct in missing_pct.items():
        print(f"  {col}: {pct:.1f}%")

    return df_changes, change_cols

def filter_date_range(df, start_date='2010-01-01', end_date='2019-12-31'):
    """Filter data to match SP500 training date range"""
    print(f"Filtering data to range: {start_date} to {end_date}")

    df_filtered = df.copy()
    df_filtered = df_filtered[
        (df_filtered['Date'] >= start_date) &
        (df_filtered['Date'] <= end_date)
    ].copy()

    print(f"Date range after filtering: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    print(f"Filtered data shape: {df_filtered.shape}")

    return df_filtered

def prepare_training_data(df, change_cols, test_size=0.2):
    """Prepare data for DistVAE training with proper handling of missing values"""
    print("Preparing training data...")

    # Only use daily change columns as continuous features
    continuous_features = change_cols

    # No categorical features
    categorical_features = []

    # Only keep rows with sufficient data (at least 70% of yield change data available)
    min_changes_required = int(0.7 * len(change_cols))
    df_clean = df.copy()
    df_clean['change_count'] = df_clean[change_cols].count(axis=1)
    df_clean = df_clean[df_clean['change_count'] >= min_changes_required].copy()

    print(f"After filtering: {len(df_clean)} rows ({len(df) - len(df_clean)} removed due to missing data)")

    # For daily changes, missing values are more problematic
    # Use forward fill sparingly, then drop remaining NaN rows
    df_clean[continuous_features] = df_clean[continuous_features].fillna(method='ffill', limit=3)

    # Drop rows with remaining missing values (daily changes should be complete)
    df_clean = df_clean.dropna(subset=continuous_features).copy()

    print(f"After dropping NaN changes: {len(df_clean)} rows")

    # Create train/test split based on time (more realistic for time series)
    n_total = len(df_clean)
    n_train = int(n_total * (1 - test_size))

    train_df = df_clean.iloc[:n_train].copy()
    test_df = df_clean.iloc[n_train:].copy()

    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    return train_df, test_df, continuous_features, categorical_features

def plot_feature_distributions(train_df, continuous_features):
    """Plot distribution of each training feature (daily yield changes)"""
    print("Creating feature distribution plots...")

    n_features = len(continuous_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(continuous_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Get data and remove NaN
        data = train_df[feature].dropna()

        # Create histogram
        ax.hist(data, bins=50, alpha=0.7, density=True, color='steelblue', edgecolor='black', linewidth=0.5)

        # Add statistics text
        mean_val = data.mean()
        std_val = data.std()
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)

        # Add normal distribution overlay for comparison
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_dist = stats.norm.pdf(x_range, mean_val, std_val)
        ax.plot(x_range, normal_dist, 'r-', linewidth=2, label='Normal fit')

        # Format maturity name for title
        maturity = feature.replace('_change', '')
        ax.set_title(f'{maturity} Daily Changes\nMean: {mean_val:.4f}, Std: {std_val:.4f}')
        ax.set_xlabel('Daily Change (bp)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics box
        stats_text = f'Skew: {skew_val:.3f}\nKurt: {kurt_val:.3f}\nN: {len(data):,}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Distribution of Daily Yield Changes by Maturity (Training Data)',
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('data/ir/yield_change_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Summary statistics table
    print(f"\n{'='*80}")
    print("DAILY YIELD CHANGE STATISTICS (Training Data)")
    print(f"{'='*80}")
    print(f"{'Maturity':<8} {'Mean':<8} {'Std':<8} {'Skew':<8} {'Kurt':<8} {'Min':<8} {'Max':<8} {'N':<8}")
    print(f"{'-'*80}")

    for feature in continuous_features:
        data = train_df[feature].dropna()
        maturity = feature.replace('_change', '')

        mean_val = data.mean()
        std_val = data.std()
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)
        min_val = data.min()
        max_val = data.max()
        count = len(data)

        print(f"{maturity:<8} {mean_val:<8.4f} {std_val:<8.4f} {skew_val:<8.3f} {kurt_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f} {count:<8}")

    print(f"{'-'*80}")
    print("Notes:")
    print("- Mean should be close to 0 (no systematic drift)")
    print("- Skew measures asymmetry (0 = symmetric)")
    print("- Kurt measures tail heaviness (0 = normal, >0 = fat tails)")
    print("- Daily changes typically show fat tails (high kurtosis)")

    print(f"Distribution plots saved as 'data/ir/yield_change_distributions.png'")

    return True

def train_ir_distvae(train_df, continuous_features, categorical_features,
                     latent_dim=6, step_size=0.05, epochs=800, beta=0.4):
    """Train DistVAE model on interest rate data"""
    print(f"\n{'='*80}")
    print(f"Training IR DistVAE - Latent: {latent_dim}, Step: {step_size}, Beta: {beta}")
    print(f"{'='*80}")

    # Prepare tensor data
    X_tensor, categorical_dims, meta = prepare_tabular_data(
        train_df, continuous_features, categorical_features, standardize=True
    )

    print(f"Prepared tensor shape: {X_tensor.shape}")
    print(f"Continuous features: {len(continuous_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Categorical dimensions: {categorical_dims}")

    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DistVAE(
        continuous_dim=len(continuous_features),
        categorical_dims=categorical_dims if categorical_dims else None,
        latent_dim=latent_dim,
        step=step_size,
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    print(f"Training for {epochs} epochs...")
    start_time = datetime.now()

    history = fit_model(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=1e-3,
        beta=beta,
        threshold=1e-6,
        device=device,
        verbose=True
    )

    training_time = datetime.now() - start_time
    print(f"Training completed in: {training_time}")

    return model, history, meta

def evaluate_ir_model(model, meta, train_df, continuous_features, change_cols):
    """Evaluate the trained IR model"""
    print("\nEvaluating IR model...")

    # Generate synthetic data
    n_samples = min(2000, len(train_df))
    with torch.no_grad():
        synthetic_tensor = model.generate(n=n_samples, return_onehot=False)

    synthetic_df = dataframe_from_samples(
        synthetic_tensor, meta, onehot_input=False, map_to_labels=True
    )

    # Focus evaluation on daily yield changes
    print(f"\nDaily Yield Change Quality Assessment:")
    print("-" * 80)
    print(f"{'Maturity':<8} {'Mean Err':<10} {'Std Err':<10} {'Skew Err':<10} {'Kurt Err':<10} {'Corr(Real)':<12} {'Corr(Synth)':<12}")
    print("-" * 80)

    results = {}
    benchmark_col = '10Y_change'  # Use 10Y changes as benchmark for correlation

    for col in change_cols:
        if col in synthetic_df.columns and col in train_df.columns:
            real_data = train_df[col].dropna()
            synth_data = synthetic_df[col].dropna()

            # Statistical moments
            mean_err = abs(real_data.mean() - synth_data.mean())
            std_err = abs(real_data.std() - synth_data.std())
            skew_err = abs(stats.skew(real_data) - stats.skew(synth_data))
            kurt_err = abs(stats.kurtosis(real_data) - stats.kurtosis(synth_data))

            # Correlation with 10Y changes (benchmark)
            real_corr = train_df[col].corr(train_df[benchmark_col]) if col != benchmark_col else 1.0
            synth_corr = synthetic_df[col].corr(synthetic_df[benchmark_col]) if col != benchmark_col else 1.0

            maturity = col.replace('_change', '')
            print(f"{maturity:<8} {mean_err:<10.4f} {std_err:<10.4f} {skew_err:<10.4f} {kurt_err:<10.4f} {real_corr:<12.4f} {synth_corr:<12.4f}")

            results[col] = {
                'mean_error': mean_err,
                'std_error': std_err,
                'skew_error': skew_err,
                'kurt_error': kurt_err,
                'real_corr_10y': real_corr,
                'synth_corr_10y': synth_corr
            }

    # Daily change correlation analysis
    print(f"\nDaily Change Correlation Analysis:")
    print("Note: Daily changes should have lower correlations than levels")

    # Sample a few key correlations
    key_pairs = [('2Y_change', '10Y_change'), ('3M_change', '10Y_change'), ('10Y_change', '30Y_change')]

    for col1, col2 in key_pairs:
        if col1 in train_df.columns and col2 in train_df.columns:
            real_corr = train_df[col1].corr(train_df[col2])
            synth_corr = synthetic_df[col1].corr(synthetic_df[col2]) if col1 in synthetic_df.columns and col2 in synthetic_df.columns else np.nan

            pair_name = f"{col1.replace('_change', '')}-{col2.replace('_change', '')}"
            print(f"  {pair_name}: Real={real_corr:.3f}, Synthetic={synth_corr:.3f}, Error={abs(real_corr-synth_corr):.3f}")

    return results, synthetic_df

def create_prediction_function(model, meta, continuous_features, change_cols):
    """Create a function to predict missing interest rates"""
    def predict_missing_rates(partial_yields, target_date=None):
        """
        Predict missing interest rates given partial yield curve data

        Args:
            partial_yields: dict with maturity -> rate (e.g., {'3M': 2.5, '10Y': 4.0})
            target_date: optional date for time-based features

        Returns:
            dict with all maturities predicted
        """
        # Create input vector
        input_data = {}

        # Fill known yield changes (note: this function now expects daily changes, not levels)
        for change_col in change_cols:
            maturity = change_col.replace('_change', '')
            input_data[change_col] = partial_yields.get(maturity, np.nan)

        # No additional features needed

        # Create DataFrame and prepare tensor
        input_df = pd.DataFrame([input_data])

        # Fill any missing daily change values with zero (no change)
        for col in continuous_features:
            if pd.isna(input_df[col].iloc[0]):
                input_df[col] = 0.0  # reasonable default for daily change

        # Prepare tensor
        X_tensor, _, _ = prepare_tabular_data(
            input_df, continuous_features, [], standardize=True
        )

        # Use model to encode and then generate multiple samples
        with torch.no_grad():
            model.eval()
            # Get latent representation of input
            z, mean, logvar = model.encode(X_tensor.to(model.device), deterministic=True)

            # Generate samples from latent space (add some noise for uncertainty)
            n_samples = 10
            samples = []
            for _ in range(n_samples):
                z_sample = mean + torch.randn_like(mean) * 0.1  # Small noise
                gamma, beta, logit = model.quantile_parameter(z_sample)

                # Sample from quantile functions
                sample_data = []
                for j in range(model.CRPS_dim):
                    alpha = torch.rand(1, 1, device=model.device)
                    sample_data.append(model.quantile_function(alpha, gamma, beta, j))

                # Handle categorical if present
                if model.softmax_dim > 0:
                    for k, dim in enumerate(model.categorical_dims):
                        logits_k = logit[:, sum(model.categorical_dims[:k]):sum(model.categorical_dims[:k+1])]
                        probs = torch.softmax(logits_k, dim=1)
                        sample_data.append(torch.multinomial(probs, 1).float())

                sample_tensor = torch.cat(sample_data, dim=1)
                samples.append(sample_tensor)

            # Average the samples
            avg_sample = torch.mean(torch.stack(samples), dim=0)

        # Convert back to DataFrame
        result_df = dataframe_from_samples(
            avg_sample, meta, onehot_input=False, map_to_labels=True
        )

        # Extract daily change predictions
        predicted_changes = {}
        for change_col in change_cols:
            if change_col in result_df.columns:
                maturity = change_col.replace('_change', '')
                predicted_changes[maturity] = float(result_df[change_col].iloc[0])

        return predicted_changes

    return predict_missing_rates

def create_visualization(train_df, synthetic_df, change_cols, results):
    """Create visualization comparing real vs synthetic daily yield changes"""
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Volatility comparison across maturities
    maturities = [col.replace('_change', '') for col in change_cols]
    real_vols = [train_df[col].std() for col in change_cols]
    synth_vols = [synthetic_df[col].std() for col in change_cols if col in synthetic_df.columns]

    maturity_years = [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]  # Approximate years for plotting

    axes[0,0].plot(maturity_years, real_vols, 'b-o', label='Real Volatility', linewidth=2, markersize=6)
    if len(synth_vols) == len(real_vols):
        axes[0,0].plot(maturity_years, synth_vols, 'r--s', label='Synthetic Volatility', linewidth=2, markersize=6)
    axes[0,0].set_xlabel('Maturity (Years)')
    axes[0,0].set_ylabel('Daily Change Volatility (pp)')
    axes[0,0].set_title('Volatility Structure Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xscale('log')

    # 2. Distribution comparison for 10Y changes
    benchmark_col = '10Y_change'
    if benchmark_col in train_df.columns and benchmark_col in synthetic_df.columns:
        axes[0,1].hist(train_df[benchmark_col].dropna(), bins=50, alpha=0.7, label='Real', density=True, color='blue')
        axes[0,1].hist(synthetic_df[benchmark_col].dropna(), bins=50, alpha=0.7, label='Synthetic', density=True, color='red')
        axes[0,1].set_xlabel('10Y Daily Change (pp)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('10Y Daily Change Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

    # 3. Correlation matrix comparison
    available_cols = [col for col in change_cols if col in train_df.columns and col in synthetic_df.columns]
    if len(available_cols) > 1:
        real_corr = train_df[available_cols].corr()
        synth_corr = synthetic_df[available_cols].corr()

        # Plot correlation difference
        corr_diff = np.abs(real_corr - synth_corr)
        im = axes[1,0].imshow(corr_diff.values, cmap='Reds', vmin=0, vmax=0.3)

        # Format labels
        short_labels = [col.replace('_change', '') for col in available_cols]
        axes[1,0].set_xticks(range(len(available_cols)))
        axes[1,0].set_yticks(range(len(available_cols)))
        axes[1,0].set_xticklabels(short_labels, rotation=45)
        axes[1,0].set_yticklabels(short_labels)
        axes[1,0].set_title('Correlation Error |Real - Synthetic|')
        plt.colorbar(im, ax=axes[1,0])

    # 4. Quality metrics summary
    metrics = ['mean_error', 'std_error', 'skew_error', 'kurt_error']
    changes_plot = [col for col in change_cols if col in results]
    maturity_labels = [col.replace('_change', '') for col in changes_plot]

    if changes_plot:
        metric_values = {metric: [results[col][metric] for col in changes_plot] for metric in metrics}

        x = np.arange(len(changes_plot))
        width = 0.2

        for i, metric in enumerate(metrics):
            axes[1,1].bar(x + i*width, metric_values[metric], width, label=metric.replace('_', ' ').title())

        axes[1,1].set_xlabel('Maturity')
        axes[1,1].set_ylabel('Error')
        axes[1,1].set_title('Quality Metrics by Maturity (Daily Changes)')
        axes[1,1].set_xticks(x + width * 1.5)
        axes[1,1].set_xticklabels(maturity_labels, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/ir/ir_distvae_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualization saved as 'data/ir/ir_distvae_analysis.png'")

def main():
    """Main execution function"""
    print("=== Interest Rate DistVAE Training ===")
    print(f"Start time: {datetime.now()}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train DistVAE on interest rate data')
    parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=6, help='Latent space dimension')
    parser.add_argument('--step_size', type=float, default=0.05, help='Quantile step size')
    parser.add_argument('--beta', type=float, default=0.4, help='KL divergence weight')
    parser.add_argument('--quick', action='store_true', help='Quick training with fewer epochs')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 200
        print("🚀 Quick mode: Using 200 epochs")

    # Ensure output directory
    os.makedirs('data/ir', exist_ok=True)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and prepare data
    df, maturity_cols = load_and_clean_ir_data()
    df_filtered = filter_date_range(df, start_date='2010-01-01', end_date='2019-12-31')
    df_changes, change_cols = convert_to_daily_changes(df_filtered, maturity_cols)
    train_df, test_df, continuous_features, categorical_features = prepare_training_data(
        df_changes, change_cols
    )

    # Plot feature distributions
    plot_feature_distributions(train_df, continuous_features)

    # Train model
    model, history, meta = train_ir_distvae(
        train_df, continuous_features, categorical_features,
        latent_dim=args.latent_dim,
        step_size=args.step_size,
        epochs=args.epochs,
        beta=args.beta
    )

    # Evaluate model
    results, synthetic_df = evaluate_ir_model(
        model, meta, train_df, continuous_features, change_cols
    )

    # Create prediction function
    predict_function = create_prediction_function(model, meta, continuous_features, change_cols)

    # Save results
    print(f"\nSaving results...")
    torch.save(model.state_dict(), 'data/ir/ir_distvae_model.pth')
    synthetic_df.to_csv('data/ir/ir_synthetic_yields.csv', index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('data/ir/ir_training_history.csv', index=False)

    # Save metadata and model config
    model_config = {
        'continuous_features': continuous_features,
        'categorical_features': categorical_features,
        'change_cols': change_cols,
        'latent_dim': args.latent_dim,
        'step_size': args.step_size,
        'beta': args.beta,
        'epochs': args.epochs
    }

    with open('data/ir/ir_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    # Create visualizations
    create_visualization(train_df, synthetic_df, change_cols, results)

    # Demo prediction
    print(f"\n🎯 DEMO: Predicting missing rates")
    print("="*50)

    # Example 1: Given some daily changes, predict others
    partial_changes_1 = {'3M': 0.05, '10Y': -0.02, '30Y': 0.01}  # Daily changes in percentage points
    predicted_1 = predict_function(partial_changes_1, '2024-06-15')
    print(f"Given daily changes: {partial_changes_1}")
    print(f"Predicted daily changes:")
    for mat in ['6M', '1Y', '2Y', '3Y', '5Y', '7Y', '20Y']:
        if mat in predicted_1:
            print(f"  {mat}: {predicted_1[mat]:.4f}pp")

    # Example 2: Given only 10Y change, predict all others
    partial_changes_2 = {'10Y': -0.15}  # 15bp decrease
    predicted_2 = predict_function(partial_changes_2, '2024-12-01')
    print(f"\nGiven daily change: {partial_changes_2}")
    print(f"Predicted daily changes:")
    change_maturities = [col.replace('_change', '') for col in change_cols]
    for mat in change_maturities:
        if mat in predicted_2 and mat != '10Y':
            print(f"  {mat}: {predicted_2[mat]:.4f}pp")

    print(f"\n✅ Training complete!")
    print(f"📁 Files saved in data/ir/:")
    print(f"   - ir_distvae_model.pth (model weights)")
    print(f"   - ir_synthetic_yields.csv (synthetic data)")
    print(f"   - ir_training_history.csv (training metrics)")
    print(f"   - ir_model_config.json (model configuration)")
    print(f"   - ir_distvae_analysis.png (analysis plots)")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()