"""g2 (Criterion b, hierarchical unbundling) inverse-relationship diagnosis.

The inverse lift ratio is 0.51-0.57, the direction is reversed. This script checks:
1. Confounder stratification: group by cluster size, number of modules, same-layer share, number of hierarchy levels; see whether the inverse relationship is consistent
2. Mechanism intermediate links: check whether clusters with holds=1 show intermediate evidence of "absorption by the upper level"
3. Label plausibility: check the distribution and features of the split labels
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(r"D:\harness office\research\jobtask-agent")
OBJ = BASE / "out" / "objects"
RES = BASE / "out" / "results"

key, outcome = "seed", "split"


def ttest(a, b):
    """Simple t test (normal-approximation p value), no scipy dependency."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va = a.var()
    vb = b.var()
    if va == 0 and vb == 0:
        return float("nan"), float("nan")
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)


# ---------- Load ----------
pool = pd.read_parquet(OBJ / "g2_comm_2020.parquet")
j = pd.read_parquet(OBJ / "judge_g2_2020.parquet")
lab = pd.read_parquet(OBJ / "labels_g2_2020.parquet")

for df in (pool, j, lab):
    df["seed"] = df["seed"].astype(str)

m = (j.merge(lab, on="seed", how="inner")
       .merge(pool[["seed", "size", "n_mod", "e2_share", "hier_n"]], on="seed", how="left"))
print(f"After merge n={len(m)}")
print()

# ---------- 1. Confounder stratification ----------
print("=" * 70)
print("1. Confounder stratification analysis")
print("=" * 70)

confound_vars = ["size", "n_mod", "e2_share", "hier_n"]

for var in confound_vars:
    print(f"\n--- Stratified by {var} ---")
    qcut = pd.qcut(m[var], 3, labels=["low", "mid", "high"])
    for label, g in m.groupby(qcut):
        a = g[g["holds"] == 1]
        b = g[g["holds"] == 0]
        p1 = a[outcome].mean() if len(a) else float("nan")
        p0 = b[outcome].mean() if len(b) else float("nan")
        lift = round(p1 / p0, 2) if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")
        print(f"  {label}: n={len(g)}, holds rate={g['holds'].mean():.3f}, "
              f"P(outcome|holds=1)={p1:.4f}, P(outcome|holds=0)={p0:.4f}, lift ratio={lift}")

# ---------- 2. Mechanism intermediate links ----------
print()
print("=" * 70)
print("2. Test of mechanism intermediate links")
print("=" * 70)
print("Criterion b chain: cohesive cluster (holds=1) → absorption by the upper level → splitting in the lower level")

print("\n--- Cluster features: holds=1 vs holds=0 ---")
for var in ["size", "n_mod", "e2_share", "hier_n"]:
    h1 = m[m["holds"] == 1][var]
    h0 = m[m["holds"] == 0][var]
    t, p = ttest(h1, h0)
    print(f"  {var}: holds=1 mean={h1.mean():.4f} (n={len(h1)}), "
          f"holds=0 mean={h0.mean():.4f} (n={len(h0)}), t={t:.3f}, p={p:.4f}")

# ---------- 3. Label plausibility ----------
print()
print("=" * 70)
print("3. Distribution and features of the split label")
print("=" * 70)
print(f"Overall split rate: {m[outcome].mean():.4f}")
print("Grouped by h:")
for h, g in m.groupby("h"):
    print(f"  h={h}: split rate={g[outcome].mean():.4f}, n={len(g)}")

print("\n--- Cluster features: split=1 vs split=0 ---")
for var in ["size", "n_mod", "e2_share", "hier_n"]:
    sp = m[m[outcome] == 1][var]
    ns = m[m[outcome] == 0][var]
    t, p = ttest(sp, ns)
    print(f"  {var}: split=1 mean={sp.mean():.4f} (n={len(sp)}), "
          f"split=0 mean={ns.mean():.4f} (n={len(ns)}), t={t:.3f}, p={p:.4f}")

# ---------- 4. holds × outcome cross-tabulation ----------
print()
print("=" * 70)
print("4. holds × outcome cross-tabulation (by h)")
print("=" * 70)
for h, g in m.groupby("h"):
    ct = pd.crosstab(g["holds"], g[outcome], margins=True)
    print(f"\nh={h}:")
    print(ct.to_string())

# ---------- 5. Regression: effect of holds after controlling for confounders ----------
print()
print("=" * 70)
print("5. Logistic regression (effect of holds after controlling for confounders)")
print("=" * 70)
# Use a simple Wald test, no statsmodels dependency
# y = split, X = [holds, size, n_mod, e2_share, hier_n]
# Standardize variables
feat_names = ["holds", "size", "n_mod", "e2_share", "hier_n"]
X = m[feat_names].copy()
for c in feat_names:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
y = m[outcome].values

# Add constant
X = np.column_stack([np.ones(len(X)), X.values])

try:
    # Logistic regression (Newton-Raphson, hand-implemented)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        # Avoid division by zero
        W = np.clip(W, 1e-10, None)
        grad = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        try:
            delta = np.linalg.solve(H + 1e-8 * np.eye(H.shape[0]), grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-8:
            break

    # Covariance matrix
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.diag(cov))

    print("variable       coef      std err    z       p-value")
    print("-" * 50)
    for i, name in enumerate(["const"] + feat_names):
        z = beta[i] / se[i] if se[i] > 0 else float("nan")
        pval = math.erfc(abs(z) / math.sqrt(2))
        sign = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{name:14s}  {beta[i]:8.4f}  {se[i]:8.4f}  {z:8.3f}  {pval:.4f} {sign}")
    print(f"\nThe coefficient on holds is {beta[1]:.4f}, meaning that for holds=1 vs holds=0,")
    print(f"odds ratio = {math.exp(beta[1]):.4f}")
    if math.exp(beta[1]) < 1:
        print("i.e., after controlling for confounders, the split odds for holds=1 are still lower than for holds=0; the inverse relationship persists.")
    else:
        print("i.e., after controlling for confounders, the split odds for holds=1 are higher than for holds=0; the inverse relationship disappears.")
except Exception as e:
    print(f"Regression failed: {e}")
