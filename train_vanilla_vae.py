#!/usr/bin/env python3
"""
Vanilla VAE Training Script for S&P 500 Comparison

This script trains a standard Vanilla VAE (with Gaussian decoder) on the same
S&P 500 data for comparison against DistVAE. Uses beta=0.1 for better reconstruction.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class VanillaVAE(nn.Module):
    """Standard VAE with Gaussian encoder/decoder for comparison with DistVAE"""

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
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(self.device)

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variables to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, n_samples):
        """Generate samples from the model"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.decode(z)
        return samples

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    """VAE loss function with reconstruction and KL divergence terms"""
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

def load_sp500_data():
    """Load and prepare S&P 500 data"""
    print("Loading S&P 500 cross-sectional data for Vanilla VAE...")

    train_df = pd.read_csv('data/sp500/sp500_cross_sectional_train.csv', index_col=0)

    # Load metadata
    with open('data/sp500/datasets_metadata.json', 'r') as f:
        metadata = json.load(f)

    stock_cols = metadata['cross_sectional_stocks']
    market_cols = ['log_return_mean', 'log_return_std', 'volume_log_mean', 'high_low_ratio_mean']
    continuous = stock_cols + market_cols

    # Standardize the data (same as DistVAE preprocessing)
    data = train_df[continuous].values

    # Z-score normalization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_normalized = (data - mean) / (std + 1e-8)

    # Store normalization parameters for later use
    normalization_params = {'mean': mean, 'std': std}

    print(f"Data shape: {data_normalized.shape}")
    print(f"Features: {len(continuous)} ({len(stock_cols)} stocks + {len(market_cols)} market)")

    return data_normalized, continuous, normalization_params, train_df

def train_vanilla_vae(data, latent_dim=32, epochs=1500, beta=0.1, learning_rate=1e-3):
    """Train Vanilla VAE on S&P 500 data"""
    print(f"\n{'='*80}")
    print(f"Training Vanilla VAE - Latent: {latent_dim}, Beta: {beta}")
    print(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    data_tensor = torch.FloatTensor(data).to(device)

    # Create data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize model
    input_dim = data.shape[1]
    model = VanillaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256],
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    history = []
    model.train()

    print(f"Training for {epochs} epochs...")
    start_time = datetime.now()

    for epoch in range(epochs):
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0

        for batch_idx, (batch_data,) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(batch_data)

            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, batch_data, mu, logvar, beta=beta
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()

        # Average losses for epoch
        n_batches = len(dataloader)
        avg_total_loss = total_loss_epoch / n_batches
        avg_recon_loss = recon_loss_epoch / n_batches
        avg_kl_loss = kl_loss_epoch / n_batches

        history.append({
            'epoch': epoch + 1,
            'total_loss': avg_total_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        })

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Recon: {avg_recon_loss:.4f}, "
                  f"KL: {avg_kl_loss:.4f}")

    training_time = datetime.now() - start_time
    print(f"Training completed in: {training_time}")

    return model, history

def generate_synthetic_samples(model, n_samples, normalization_params, continuous_cols):
    """Generate synthetic samples and denormalize them"""
    print(f"Generating {n_samples} synthetic samples...")

    model.eval()
    with torch.no_grad():
        # Generate samples
        synthetic_tensor = model.generate(n_samples)
        synthetic_data = synthetic_tensor.cpu().numpy()

    # Denormalize
    mean = normalization_params['mean']
    std = normalization_params['std']
    synthetic_denormalized = synthetic_data * std + mean

    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_denormalized, columns=continuous_cols)

    print(f"Generated {len(synthetic_df)} synthetic samples")
    return synthetic_df

def quick_quality_evaluation(model, real_df, continuous_cols, normalization_params):
    """Quick evaluation of Vanilla VAE quality"""
    print("Performing quick quality evaluation...")

    # Generate synthetic data
    synthetic_df = generate_synthetic_samples(
        model, len(real_df), normalization_params, continuous_cols
    )

    # Select stocks for evaluation
    stock_cols = [col for col in continuous_cols if not col.startswith('log_return_')][:10]

    print(f"\nVanilla VAE Quality Assessment:")
    print("-" * 70)
    print(f"{'Stock':<8} {'Mean Err':<10} {'Std Err':<10} {'Skew Err':<10} {'Kurt Err':<10}")
    print("-" * 70)

    metrics = []
    for col in stock_cols:
        real_data = real_df[col].dropna()
        synth_data = synthetic_df[col].dropna()

        # Calculate errors
        mean_err = abs(real_data.mean() - synth_data.mean())
        std_err = abs(real_data.std() - synth_data.std())

        from scipy import stats
        skew_err = abs(stats.skew(real_data) - stats.skew(synth_data))
        kurt_err = abs(stats.kurtosis(real_data) - stats.kurtosis(synth_data))

        print(f"{col:<8} {mean_err:<10.4f} {std_err:<10.4f} {skew_err:<10.4f} {kurt_err:<10.4f}")

        metrics.append({
            'stock': col,
            'mean_err': mean_err,
            'std_err': std_err,
            'skew_err': skew_err,
            'kurt_err': kurt_err
        })

    # Calculate averages
    avg_mean_err = np.mean([m['mean_err'] for m in metrics])
    avg_std_err = np.mean([m['std_err'] for m in metrics])
    avg_skew_err = np.mean([m['skew_err'] for m in metrics])
    avg_kurt_err = np.mean([m['kurt_err'] for m in metrics])

    print("-" * 70)
    print(f"{'AVERAGE':<8} {avg_mean_err:<10.4f} {avg_std_err:<10.4f} {avg_skew_err:<10.4f} {avg_kurt_err:<10.4f}")

    # Correlation analysis
    stock_subset = stock_cols
    real_corr = real_df[stock_subset].corr()
    synth_corr = synthetic_df[stock_subset].corr()
    corr_diff = np.abs(real_corr - synth_corr)

    upper_tri_mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    avg_corr_error = corr_diff.values[upper_tri_mask].mean()

    print(f"\nAverage correlation error: {avg_corr_error:.4f}")

    return synthetic_df, {
        'avg_mean_err': avg_mean_err,
        'avg_std_err': avg_std_err,
        'avg_skew_err': avg_skew_err,
        'avg_kurt_err': avg_kurt_err,
        'avg_corr_error': avg_corr_error
    }

def save_vanilla_vae_results(model, history, synthetic_df, quality_metrics, normalization_params):
    """Save Vanilla VAE model and results"""
    print("Saving Vanilla VAE results...")

    # Save model
    model_path = 'data/sp500/vanilla_vae_model_beta0.1.pth'
    torch.save(model.state_dict(), model_path)

    # Save synthetic data
    synthetic_df.to_csv('data/sp500/vanilla_vae_synthetic_beta0.1.csv', index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('data/sp500/vanilla_vae_history_beta0.1.csv', index=False)

    # Save normalization parameters and quality metrics
    results = {
        'model_config': {
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'beta': 0.1
        },
        'normalization_params': {
            'mean': normalization_params['mean'].tolist(),
            'std': normalization_params['std'].tolist()
        },
        'quality_metrics': quality_metrics,
        'timestamp': datetime.now().isoformat()
    }

    with open('data/sp500/vanilla_vae_results_beta0.1.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Vanilla VAE results saved:")
    print(f"   - Model: {model_path}")
    print(f"   - Synthetic data: data/sp500/vanilla_vae_synthetic_beta0.1.csv")
    print(f"   - Training history: data/sp500/vanilla_vae_history_beta0.1.csv")
    print(f"   - Results: data/sp500/vanilla_vae_results_beta0.1.json")

def plot_training_history(history):
    """Plot Vanilla VAE training history"""
    print("Creating training history plot...")

    epochs = [h['epoch'] for h in history]
    total_losses = [h['total_loss'] for h in history]
    recon_losses = [h['recon_loss'] for h in history]
    kl_losses = [h['kl_loss'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total loss
    axes[0].plot(epochs, total_losses, 'b-', linewidth=2)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, recon_losses, 'g-', linewidth=2)
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

    # KL loss
    axes[2].plot(epochs, kl_losses, 'r-', linewidth=2)
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/sp500/vanilla_vae_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Training history plot saved: data/sp500/vanilla_vae_training_history.png")

def main():
    """Main execution for Vanilla VAE training"""
    print("=== Vanilla VAE Training for DistVAE Comparison ===")
    print(f"Start time: {datetime.now()}")

    # Ensure output directory
    os.makedirs('data/sp500', exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and prepare data
    data, continuous_cols, normalization_params, real_df = load_sp500_data()

    # Train Vanilla VAE
    model, history = train_vanilla_vae(
        data,
        latent_dim=32,  # Same as best DistVAE
        epochs=1500,    # Same as DistVAE
        beta=0.1,       # Lower beta for better reconstruction
        learning_rate=1e-3
    )

    # Evaluate quality
    synthetic_df, quality_metrics = quick_quality_evaluation(
        model, real_df, continuous_cols, normalization_params
    )

    # Save results
    save_vanilla_vae_results(model, history, synthetic_df, quality_metrics, normalization_params)

    # Plot training history
    plot_training_history(history)

    print(f"\n🏆 VANILLA VAE TRAINING SUMMARY:")
    print(f"{'='*60}")
    print(f"Configuration: Latent=32, Beta=0.1, Epochs=1500")
    print(f"Average Kurtosis Error: {quality_metrics['avg_kurt_err']:.3f}")
    print(f"Average Correlation Error: {quality_metrics['avg_corr_error']:.4f}")
    print(f"Average Mean Error: {quality_metrics['avg_mean_err']:.4f}")
    print(f"Average Std Error: {quality_metrics['avg_std_err']:.4f}")

    print(f"\n💡 Next step: Run comparison script to analyze DistVAE vs Vanilla VAE")
    print(f"✅ Vanilla VAE training complete!")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()