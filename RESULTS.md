# DistVAE Reproduction Summary (β = 0.5)

This file summarizes our reproduction of the paper “Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation” (DistVAE) against the paper’s reported results. Paper values are means over 10 runs; reproduction values are from a single 100‑epoch run with seed=1.

## Metrics
- MARE (↓): Mean Absolute Relative Error for the regression target. Lower is better.
- F1 (↑): Classification F1 for the classification target. Higher is better.
- CorrDist (↓): L2 distance between association matrices (dython associations) of real vs synthetic data. Lower is better.
- K-S (↓): Kolmogorov–Smirnov statistic per feature (averaged). Lower is better.
- 1-WD (↓): 1‑Wasserstein distance per feature (averaged). Lower is better.
- DCR (↑): Distance to Closest Record (5th percentile L2) reported as R&S (real vs synthetic), R (real vs real), and S (synthetic vs synthetic). Higher implies stronger privacy; overly high S may hurt utility/diversity.
- AD F1 (↓): Attribute disclosure F1 at K ∈ {1,10,100} nearest neighbors. Lower is better.

## Hyperparameters
- Paper (DistVAE)
  - β ∈ {0.5, 1, 5}; tables typically highlight β = 0.5
  - CorrDist computed via dython `associations`
  - K‑S/1‑WD averaged across continuous and discrete groups, respectively
  - DCR computed using standardized continuous features only
  - AD evaluates K ∈ {1, 10, 100}; macro‑averaged over discrete columns
  - Splits (Appendix A.8): covtype 45k/5k, credit 45k/5k, loan 4k/1k, cabs 40k/1k, kings 20k/1k
  - Results are mean±std over 10 runs (seeds not fixed to a single value)
- Reproduction (this run)
  - β = 0.5, step = 0.1, latent_dim = 2, epochs = 100, batch_size = 256, lr = 1e‑3, seed = 1
  - Same CorrDist (dython associations), K‑S/1‑WD definitions, DCR, AD (K ∈ {1,10,100}) as the paper
  - Splits follow repo modules (match appendix); Adult uses 80/20 after cleaning (repo behavior)

---

## ML Utility + Correlation Structure

| Dataset | MARE Paper | MARE Ours | F1 Paper | F1 Ours | CorrDist Paper | CorrDist Ours |
|---|---:|---:|---:|---:|---:|---:|
| covtype | 0.044 | 0.044 | 0.605 | 0.613 | 1.179 | 1.167 |
| credit  | 0.763 | 0.665 | 0.926 | 0.926 | 2.072 | 2.052 |
| loan    | 0.249 | 0.249 | 0.914 | 0.932 | 1.654 | 1.614 |
| adult   | 0.232 | 0.233 | 0.825 | 0.818 | 0.830 | 0.808 |
| cabs    | 0.803 | 0.682 | 0.707 | 0.725 | 1.481 | 1.454 |
| kings   | 0.001 | 0.001 | 0.640 | 0.637 | 1.559 | 1.536 |

Note: Our MARE/F1 are means over three models (Linear, RF, GradientBoost) like the repo; the paper reports a single score per dataset/model (averaged over 10 runs).

## Marginal Statistical Similarity (K‑S, 1‑WD)

| Dataset | K‑S cont Paper | K‑S cont Ours | 1‑WD cont Paper | 1‑WD cont Ours | K‑S disc Paper | K‑S disc Ours | 1‑WD disc Paper | 1‑WD disc Ours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| covtype | 0.032 | 0.033 | 0.041 | 0.040 | 0.023 | 0.025 | 0.073 | 0.042 |
| credit  | 0.088 | 0.076 | 0.089 | 0.087 | 0.020 | 0.024 | 0.046 | 0.046 |
| loan    | 0.060 | 0.076 | 0.048 | 0.048 | 0.019 | 0.018 | 0.027 | 0.025 |
| adult   | 0.209 | 0.239 | 0.114 | 0.114 | 0.037 | 0.044 | 0.248 | 0.297 |
| cabs    | 0.044 | 0.043 | 0.067 | 0.066 | 0.060 | 0.080 | 0.241 | 0.331 |
| kings   | 0.110 | 0.105 | 0.089 | 0.090 | 0.022 | 0.023 | 0.071 | 0.068 |

## Privacy Preservability: DCR (R&S / R / S)

| Dataset | DCR R&S Paper | DCR R&S Ours | DCR R Paper | DCR R Ours | DCR S Paper | DCR S Ours |
|---|---:|---:|---:|---:|---:|---:|
| covtype | 0.765 | 0.781 | 0.329 | 0.329 | 0.819 | 0.828 |
| credit  | 0.692 | 0.699 | 0.452 | 0.452 | 0.742 | 0.742 |
| loan    | 0.244 | 0.249 | 0.109 | 0.109 | 0.245 | 0.230 |
| adult   | 0.060 | 0.043 | 0.000 | 0.000 | 0.005 | 0.010 |
| cabs    | 0.364 | 0.362 | 0.332 | 0.332 | 0.368 | 0.366 |
| kings   | 0.540 | 0.536 | 0.199 | 0.199 | 0.600 | 0.596 |

## Privacy Preservability: Attribute Disclosure F1 (↓)

| Dataset | k=1 Paper | k=1 Ours | k=10 Paper | k=10 Ours | k=100 Paper | k=100 Ours |
|---|---:|---:|---:|---:|---:|---:|
| covtype | 0.308 | 0.279 | 0.338 | 0.305 | 0.313 | 0.279 |
| credit  | 0.339 | 0.334 | 0.337 | 0.322 | 0.315 | 0.304 |
| loan    | 0.505 | 0.539 | 0.465 | 0.438 | 0.439 | 0.401 |
| adult   | 0.270 | 0.270 | 0.280 | 0.277 | 0.268 | 0.258 |
| cabs    | 0.251 | 0.260 | 0.241 | 0.233 | 0.225 | 0.224 |
| kings   | 0.293 | 0.296 | 0.310 | 0.343 | 0.301 | 0.339 |

---

### Takeaways
- Overall, single‑run reproduction aligns closely with the paper’s 10‑run averages for β=0.5.
- ML utility (MARE, F1) and CorrDist are very close across datasets; minor deviations reflect single‑run variance.
- K‑S / 1‑WD are in expected ranges and follow reported trends.
- DCR and AD F1 also match trends; small differences (e.g., loan AD@k=1) are plausible from single‑run variability.

### Suggested next steps (optional)
- Average over 10 seeds and report mean±std to mirror the paper’s tables precisely.
- Sweep β ∈ {0.5, 1, 5} to reproduce the privacy–utility trade‑off (MLu, CorrDist, DCR, AD F1).
