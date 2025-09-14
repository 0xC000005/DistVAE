#!/usr/bin/env python3
"""
Demo: Train DistVAE on a CSV and write a synthetic CSV.

Example (Adult dataset columns from Kaggle adult.csv):

python demo_distvae.py \
  --csv ./data/adult.csv \
  --continuous 'age,education.num,capital.gain,capital.loss,hours.per.week' \
  --discrete 'workclass,education,marital.status,occupation,relationship,race,sex,native.country,income' \
  --epochs 100 \
  --out ./assets/adult_synthetic.csv

Notes:
- Continuous columns are standardized inside; they're unstandardized before writing outputs.
- Categorical columns are one-hot encoded for training and returned as integer indices in outputs;
  pass --map-to-labels to write original string labels instead of indices.
"""

import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from distvae_standalone import (
    DistVAE,
    fit_model,
    prepare_tabular_data,
    dataframe_from_samples,
)


def _parse_list(arg: str) -> List[str]:
    if arg is None or arg.strip() == "":
        return []
    return [s.strip() for s in arg.split(',') if s.strip()]


def main():
    ap = argparse.ArgumentParser("DistVAE CSV demo")
    ap.add_argument('--csv', required=True, help='Input CSV path')
    ap.add_argument('--continuous', required=True, help='Comma-separated continuous column names')
    ap.add_argument('--discrete', required=True, help='Comma-separated categorical column names')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--latent-dim', type=int, default=2)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--n-samples', type=int, default=0, help='Synthetic rows to generate (default: len(input))')
    ap.add_argument('--out', required=True, help='Output synthetic CSV path')
    ap.add_argument('--map-to-labels', action='store_true', help='Map categorical indices back to original labels')
    args = ap.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load CSV
    df = pd.read_csv(args.csv)
    continuous = _parse_list(args.continuous)
    discrete = _parse_list(args.discrete)
    for c in continuous + discrete:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")

    # Prepare tensor and meta
    X, categorical_dims, meta = prepare_tabular_data(df, continuous, discrete, standardize=True)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = (
        torch.device('cuda:0') if (args.device == 'auto' and torch.cuda.is_available())
        else torch.device('cpu') if args.device in ['auto', 'cpu'] else torch.device('cuda:0')
    )

    # Model
    model = DistVAE(
        continuous_dim=len(continuous),
        categorical_dims=categorical_dims,
        latent_dim=args.latent_dim,
        step=args.step,
        device=device,
    )

    # Train
    print(f"Training DistVAE on {args.csv} (rows={len(df)}) ...")
    fit_model(
        model,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        threshold=1e-5,
        device=device,
        verbose=True,
    )

    # Generate
    n = args.n_samples if args.n_samples > 0 else len(df)
    print(f"Generating {n} synthetic rows ...")
    with torch.no_grad():
        Y = model.generate(n=n, return_onehot=False)

    # Back to DataFrame with original column names, unstandardize, and map labels if requested
    df_syn = dataframe_from_samples(Y, meta, onehot_input=False, map_to_labels=args.map_to_labels)
    df_syn.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == '__main__':
    main()

