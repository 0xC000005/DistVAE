"""
Standalone DistVAE for tabular data

This module implements the DistVAE model (distributional VAE with a quantile decoder)
as a single-file dependency you can import and use on arbitrary tabular tensors or pandas DataFrames.

Core ideas (aligned with the paper and repo implementation):
- Continuous variables are decoded via a conditional quantile function parameterized as
  a positive-slope spline over alpha in [0,1] (piecewise-linear with Softplus slopes).
- Categorical variables are decoded with a softmax head (one head per categorical group).
- Reconstruction loss: CRPS for continuous + CrossEntropy for categorical, plus beta * KL.

Quickstart A — from raw tensors
--------------------------------
from distvae_standalone import DistVAE, fit_model
import torch; from torch.utils.data import DataLoader, TensorDataset

# Define schema
continuous_dim = 10
categorical_dims = [3, 4]  # e.g., two categorical columns with 3 and 4 unique values

# Build DataLoader with standardized continuous features and one-hot categorical segments
X = torch.randn(1000, continuous_dim + sum(categorical_dims))
loader = DataLoader(TensorDataset(X.float()), batch_size=256, shuffle=True)

# Model and training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DistVAE(continuous_dim, categorical_dims, latent_dim=2, step=0.1, device=device)
fit_model(model, loader, epochs=100, lr=1e-3, beta=0.5, threshold=1e-5, device=device)

# Sampling (standardized space)
Y = model.generate(n=500, return_onehot=True)  # shape: [n, D]

Quickstart B — from pandas DataFrame
------------------------------------
from distvae_standalone import (
    DistVAE, fit_model, prepare_tabular_data, dataframe_from_samples
)
import pandas as pd; from torch.utils.data import DataLoader, TensorDataset

# Define column names
continuous = ['age', 'capital.gain', 'capital.loss']
discrete = ['workclass', 'education']
df = pd.read_csv('your.csv')

# Prepare tensor and metadata (standardizes continuous, one-hot encodes categorical)
X, categorical_dims, meta = prepare_tabular_data(df, continuous, discrete, standardize=True)
loader = DataLoader(TensorDataset(X), batch_size=256, shuffle=True)

# Train
model = DistVAE(len(continuous), categorical_dims, latent_dim=2, step=0.1)
fit_model(model, loader, epochs=100, lr=1e-3, beta=0.5)

# Generate samples and convert back to DataFrame with original column names
Y = model.generate(n=len(df), return_onehot=False)
df_syn = dataframe_from_samples(Y, meta, onehot_input=False, map_to_labels=True)
df_syn.to_csv('synthetic.csv', index=False)

CLI demo
--------
See demo_distvae.py for a CLI example that:
- Reads a CSV
- Builds a DataLoader via prepare_tabular_data
- Trains DistVAE and writes a synthetic CSV

Notes
-----
- The DataLoader must yield FloatTensor batches of shape [B, D], where D = continuous_dim + sum(categorical_dims).
- For categorical training, pass one-hot segments in the same order as categorical_dims.
- prepare_tabular_data/dataframe_from_samples help standardize/one-hot encode and map back to original labels.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistVAE(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: Optional[List[int]] = None,
        latent_dim: int = 2,
        step: float = 0.1,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        self.continuous_dim = int(continuous_dim)
        self.categorical_dims = list(categorical_dims or [])
        self.latent_dim = latent_dim
        self.step = float(step)

        self.CRPS_dim = self.continuous_dim
        self.softmax_dim = sum(self.categorical_dims)

        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device(device)

        input_dim = self.continuous_dim + self.softmax_dim

        # encoder: x -> (mean, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.latent_dim * 2),
        ).to(self.device)

        # quantile grid (delta) and spline head
        self.delta = torch.arange(0.0, 1.0 + self.step, step=self.step).view(1, -1).to(self.device)
        self.M = self.delta.size(1) - 1
        # For each continuous dim j: gamma_j (1) + beta_j (M+1), so CRPS_dim * (1 + M+1)
        spline_out = self.CRPS_dim * (1 + (self.M + 1)) + self.softmax_dim
        self.spline = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, spline_out),
        ).to(self.device)

    # ----- latent posterior -----
    def get_posterior(self, x: torch.Tensor):
        h = self.encoder(x)
        mean, logvar = torch.split(h, self.latent_dim, dim=1)
        return mean, logvar

    def sampling(self, mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False):
        if deterministic:
            return mean
        noise = torch.randn(mean.size(0), self.latent_dim, device=self.device)
        z = mean + torch.exp(logvar / 2) * noise
        return z

    def encode(self, x: torch.Tensor, deterministic: bool = False):
        mean, logvar = self.get_posterior(x)
        z = self.sampling(mean, logvar, deterministic=deterministic)
        return z, mean, logvar

    # ----- quantile parameterization -----
    def quantile_parameter(self, z: torch.Tensor):
        h = self.spline(z)
        if self.softmax_dim > 0:
            logit = h[:, -self.softmax_dim:]
            spline = h[:, :-self.softmax_dim]
        else:
            logit = torch.empty((z.size(0), 0), device=z.device)
            spline = h
        chunks = torch.split(spline, 1 + (self.M + 1), dim=1)
        gamma = [c[:, [0]] for c in chunks]  # list of [B,1]
        beta = [F.softplus(c[:, 1:]) for c in chunks]  # list of [B,M+1], positivity
        return gamma, beta, logit

    def quantile_function(self, alpha: torch.Tensor, gamma: List[torch.Tensor], beta: List[torch.Tensor], j: int):
        # Q_j(alpha; z) = gamma_j + sum_k beta_jk * max(alpha - delta_k, 0)
        # alpha: [B,1]
        return gamma[j] + (beta[j] * torch.where(
            alpha - self.delta > 0,
            alpha - self.delta,
            torch.zeros((), device=self.device)
        )).sum(dim=1, keepdims=True)

    def _quantile_inverse(self, xj: torch.Tensor, gamma: List[torch.Tensor], beta: List[torch.Tensor], j: int):
        # Invert piecewise-linear Q_j(alpha;z) for given xj
        # Build mask grid at knots: gamma_j + sum_k beta_jk * max(delta_t - delta_k, 0)
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(
            delta_ - self.delta > 0,
            delta_ - self.delta,
            torch.zeros((), device=self.device)
        )
        mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(dim=-1).squeeze(0).t()  # [B, M+1]
        mask = torch.where(mask <= xj, mask, torch.zeros((), device=self.device)).bool().float()
        alpha_tilde = xj - gamma[j]
        alpha_tilde += (mask * beta[j] * self.delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta[j]).sum(dim=1, keepdims=True) + 1e-6
        # numerical stability: clip to (epsilon, 1)
        return torch.clip(alpha_tilde, 1e-5, 1.0)

    def quantile_inverse(self, x: torch.Tensor, gamma: List[torch.Tensor], beta: List[torch.Tensor]):
        # x: [B, D_cont]
        alpha_tilde_list = []
        for j in range(self.CRPS_dim):
            alpha_tilde = self._quantile_inverse(x[:, [j]], gamma, beta, j)
            alpha_tilde_list.append(alpha_tilde)
        return alpha_tilde_list

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        z, mean, logvar = self.encode(x, deterministic=deterministic)
        gamma, beta, logit = self.quantile_parameter(z)
        return z, mean, logvar, gamma, beta, logit

    # ----- sampling -----
    @staticmethod
    def _gumbel_sampling(size, device, eps: float = 1e-20):
        U = torch.rand(size, device=device)
        G = -torch.log(-torch.log(U + eps) + eps)
        return G

    @torch.no_grad()
    def generate(self, n: int, return_onehot: bool = True) -> torch.Tensor:
        """Generate n synthetic samples (standardized space).

        Returns a tensor of shape [n, continuous_dim + sum(categorical_dims)].
        Continuous come first, then a flat concatenation of one-hot segments for each categorical.
        If return_onehot=False, categorical will be indices in a float tensor (not one-hot).
        """
        self.eval()
        device = self.device
        batch_size = min(n, 1024)
        steps = n // batch_size + int(n % batch_size != 0)

        out_batches = []
        for _ in range(steps):
            randn = torch.randn(batch_size, self.latent_dim, device=device)
            gamma, beta, logit = self.quantile_parameter(randn)

            samples = []
            # continuous
            for j in range(self.CRPS_dim):
                alpha = torch.rand(batch_size, 1, device=device)
                samples.append(self.quantile_function(alpha, gamma, beta, j))

            # categorical
            st = 0
            if self.softmax_dim > 0:
                for dim in self.categorical_dims:
                    ed = st + dim
                    logits_j = logit[:, st:ed]
                    # Gumbel-Max
                    G = self._gumbel_sampling(logits_j.shape, device=device)
                    _, idx = (F.log_softmax(logits_j, dim=1) + G).max(dim=1)
                    if return_onehot:
                        samples.append(F.one_hot(idx, num_classes=dim).float())
                    else:
                        samples.append(idx.float().unsqueeze(1))
                    st = ed

            out = torch.cat(samples, dim=1) if samples else torch.empty((batch_size, 0), device=device)
            out_batches.append(out)

        out = torch.cat(out_batches, dim=0)
        return out[:n]


# ======== Utilities for pandas DataFrames ========

def prepare_tabular_data(
    df: pd.DataFrame,
    continuous: List[str],
    discrete: List[str],
    *,
    standardize: bool = True,
) -> Tuple[torch.FloatTensor, List[int], Dict[str, Any]]:
    """Prepare tabular tensor and metadata from a pandas DataFrame.

    - continuous: names of continuous columns (will be placed first)
    - discrete: names of categorical columns (will be one-hot encoded and placed after continuous)
    - standardize: if True, z-score standardize continuous columns and store mean/std in metadata

    Returns (X_tensor, categorical_dims, meta).
    """
    df = df.copy()
    meta: Dict[str, Any] = {
        "continuous": list(continuous),
        "discrete": list(discrete),
    }

    # standardize continuous
    if standardize and len(continuous) > 0:
        mean = df[continuous].mean(axis=0)
        std = df[continuous].std(axis=0).replace(0, 1.0)
        df[continuous] = (df[continuous] - mean) / std
        meta["mean"] = mean
        meta["std"] = std
    else:
        meta["mean"] = pd.Series(np.zeros(len(continuous)), index=continuous)
        meta["std"] = pd.Series(np.ones(len(continuous)), index=continuous)

    # categorical mappings and one-hot encoding
    categorical_dims: List[int] = []
    discrete_maps: List[Dict[Any, int]] = []
    discrete_reverse: List[Dict[int, Any]] = []
    onehot_parts: List[pd.DataFrame] = []
    for d in discrete:
        cats = sorted(pd.Series(df[d]).dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(cats)}
        reverse = {i: v for v, i in mapping.items()}
        discrete_maps.append(mapping)
        discrete_reverse.append(reverse)
        idx = df[d].map(mapping).fillna(0).astype(int)
        oh = pd.get_dummies(idx, prefix=d)
        # ensure full width
        max_dim = len(cats)
        for i in range(max_dim):
            col = f"{d}_{i}"
            if col not in oh.columns:
                oh[col] = 0
        oh = oh[[f"{d}_{i}" for i in range(max_dim)]]
        onehot_parts.append(oh)
        categorical_dims.append(max_dim)

    meta["categorical_dims"] = categorical_dims
    meta["discrete_maps"] = discrete_maps
    meta["discrete_reverse"] = discrete_reverse

    # assemble feature matrix: continuous first, then one-hot blocks
    parts: List[pd.DataFrame] = []
    if len(continuous) > 0:
        parts.append(df[continuous])
    if len(onehot_parts) > 0:
        parts.extend(onehot_parts)
    X_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X_tensor = torch.from_numpy(X_df.to_numpy().astype(np.float32))
    return X_tensor, categorical_dims, meta


def dataframe_from_samples(
    samples: torch.Tensor,
    meta: Dict[str, Any],
    *,
    onehot_input: bool = False,
    map_to_labels: bool = False,
) -> pd.DataFrame:
    """Convert model samples back to a pandas DataFrame with original column names.

    - If onehot_input is False, samples are [cont | categorical indices] (one column per discrete).
    - If onehot_input is True, samples are [cont | flat one-hot segments] per meta['categorical_dims'].
    - If map_to_labels is True, map categorical indices to original labels using meta['discrete_reverse'].
    - Continuous are unstandardized using meta['mean'] and meta['std'] if present.
    """
    cont_names: List[str] = meta.get("continuous", [])
    disc_names: List[str] = meta.get("discrete", [])
    cat_dims: List[int] = meta.get("categorical_dims", [])
    mean: pd.Series = meta.get("mean", pd.Series(np.zeros(len(cont_names)), index=cont_names))
    std: pd.Series = meta.get("std", pd.Series(np.ones(len(cont_names)), index=cont_names))
    discrete_reverse: List[Dict[int, Any]] = meta.get("discrete_reverse", [{} for _ in disc_names])

    X = samples.detach().cpu().numpy()
    n = X.shape[0]
    cols: Dict[str, Any] = {}

    # continuous
    if len(cont_names) > 0:
        Xc = X[:, : len(cont_names)]
        Xc = Xc * std.values + mean.values
        for i, name in enumerate(cont_names):
            cols[name] = Xc[:, i]

    # categorical indices
    if not onehot_input:
        st = len(cont_names)
        for j, name in enumerate(disc_names):
            idx = X[:, st + j].round().astype(int)
            if map_to_labels and j < len(discrete_reverse):
                reverse = discrete_reverse[j]
                cols[name] = np.array([reverse.get(int(v), int(v)) for v in idx])
            else:
                cols[name] = idx
    else:
        st = len(cont_names)
        for j, (name, dim) in enumerate(zip(disc_names, cat_dims)):
            ed = st + dim
            oh = X[:, st:ed]
            idx = oh.argmax(axis=1).astype(int)
            if map_to_labels and j < len(discrete_reverse):
                reverse = discrete_reverse[j]
                cols[name] = np.array([reverse.get(int(v), int(v)) for v in idx])
            else:
                cols[name] = idx
            st = ed

    df_out = pd.DataFrame(cols)
    return df_out



def _crps_loss(
    x: torch.Tensor,
    model: DistVAE,
    gamma: List[torch.Tensor],
    beta: List[torch.Tensor],
    alpha_tilde_list: List[torch.Tensor],
) -> torch.Tensor:
    """Compute CRPS reconstruction loss for continuous dims (mean over batch)."""
    total_loss = x.new_zeros(())
    for j in range(model.CRPS_dim):
        alpha_tilde = alpha_tilde_list[j]
        # term from repo train.py
        term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde, model.delta).pow(2)
        term += 2 * torch.maximum(alpha_tilde, model.delta) * model.delta
        loss = (2 * alpha_tilde) * x[:, [j]]
        loss += (1 - 2 * alpha_tilde) * gamma[j]
        loss += (beta[j] * term).sum(dim=1, keepdims=True)
        loss *= 0.5
        total_loss = total_loss + loss.mean()
    return total_loss


def train_one_epoch(
    model: DistVAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    beta_kl: float = 0.5,
    threshold: float = 1e-5,
    device: Optional[torch.device | str] = None,
) -> Dict[str, float]:
    if device is None:
        device = model.device
    device = torch.device(device)

    logs = {"loss": [], "quantile": [], "KL": []}
    model.train()

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        optimizer.zero_grad()

        z, mean, logvar, gamma, beta, logit = model(x)

        # alpha_tilde for continuous dims
        if model.CRPS_dim > 0:
            alpha_tilde_list = model.quantile_inverse(x[:, : model.CRPS_dim], gamma, beta)
            # clip for numerical stability
            alpha_tilde_list = [torch.clip(a, threshold, 1.0) for a in alpha_tilde_list]
            rec_cont = _crps_loss(x[:, : model.CRPS_dim], model, gamma, beta, alpha_tilde_list)
        else:
            rec_cont = x.new_zeros(())

        # categorical cross-entropy
        rec_cat = x.new_zeros(())
        if model.softmax_dim > 0 and len(model.categorical_dims) > 0:
            st = 0
            st_in = model.CRPS_dim
            for dim in model.categorical_dims:
                ed = st + dim
                logits_j = logit[:, st:ed]
                targets_onehot = x[:, st_in + st : st_in + ed]
                targets = targets_onehot.argmax(dim=1)
                rec_cat = rec_cat + F.cross_entropy(logits_j, targets)
                st = ed

        rec = rec_cont + rec_cat

        # KL(q||p)
        KL = (mean.pow(2).sum(dim=1) - logvar.sum(dim=1) + torch.exp(logvar).sum(dim=1) - model.latent_dim)
        KL = 0.5 * KL.mean()

        loss = rec + beta_kl * KL
        loss.backward()
        optimizer.step()

        logs["loss"].append(loss.item())
        logs["quantile"].append(rec.item())
        logs["KL"].append(KL.item())

    return {k: float(sum(v) / max(1, len(v))) for k, v in logs.items()}


def fit_model(
    model: DistVAE,
    dataloader: torch.utils.data.DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 0.5,
    threshold: float = 1e-5,
    device: Optional[torch.device | str] = None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Train the given DistVAE model on a DataLoader of tabular tensors.

    The dataloader should yield a FloatTensor of shape [B, D], where D = continuous_dim + sum(categorical_dims),
    with categorical features provided as one-hot segments in the same column order used to construct the model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist: List[Dict[str, float]] = []
    for ep in range(1, epochs + 1):
        metrics = train_one_epoch(model, dataloader, optimizer, beta_kl=beta, threshold=threshold, device=device)
        hist.append(metrics)
        if verbose:
            print(f"[epoch {ep:03d}] " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return hist
