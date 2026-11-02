"""g3 (Criterion c, mimetic template diffusion) vacuous diagnosis.

The inverse lift ratio is 1.00-1.10; the net correspondence strength is nearly zero. This script checks:
1. p-bucket ordering-power test: do high-p buckets correspond to higher outcome rates (the calibration table already shows inversion)
2. Antecedent-tightening experiment: change "share above median" to stricter definitions and see whether the discriminating power recovers
3. Outcome-label replacement: from "stock above median" to "adoption increment"
4. Confounder test: does holds have an independent effect after controlling for firm characteristics
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(r"D:\harness office\research\jobtask-agent")
OBJ = BASE / "out" / "objects"

key, outcome = "firm", "above_median"


def ttest(a, b):
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
pool = pd.read_parquet(OBJ / "g3_firms_2020.parquet")
j = pd.read_parquet(OBJ / "judge_g3_2020.parquet")
lab = pd.read_parquet(OBJ / "labels_g3_2020.parquet")

for df in (pool, j, lab):
    df["firm"] = df["firm"].astype(str)

m = (j.merge(lab, on="firm", how="inner")
       .merge(pool[["firm", "n_jobs", "mean_size", "n_mod", "e2_share", "struct_sim", "xfirm_sim", "surf_sim"]],
              on="firm", how="left"))
print(f"After merge n={len(m)}")
print()

# ---------- 1. p-bucket ordering-power test ----------
print("=" * 70)
print("1. p-bucket ordering-power test")
print("=" * 70)
print("If Criterion c is valid, high-p buckets should correspond to higher outcome rates.")
print()
for h, g in m.groupby("h"):
    print(f"--- h={h} ---")
    for pval in sorted(g["p"].unique()):
        sub = g[g["p"] == pval]
        rate = sub[outcome].mean() if len(sub) else float("nan")
        print(f"  p={pval}: n={len(sub)}, actual outcome rate={rate:.4f}, holds rate={sub['holds'].mean():.3f}")
    # Correlation coefficient
    valid = g.dropna(subset=["p"])
    if len(valid) > 2:
        r = valid["p"].corr(valid[outcome], method="spearman")
        print(f"  Spearman correlation: {r:.4f}")
print()

# ---------- 2. holds × outcome cross-tabulation ----------
print("=" * 70)
print("2. holds × outcome cross-tabulation")
print("=" * 70)
for h, g in m.groupby("h"):
    ct = pd.crosstab(g["holds"], g[outcome], margins=True)
    print(f"\nh={h} (n={len(g)}):")
    print(ct.to_string())

# ---------- 3. Antecedent-tightening experiment ----------
print()
print("=" * 70)
print("3. Antecedent-tightening experiment")
print("=" * 70)
# Current antecedent: LLM-judged holds (the "mimetic template diffusion" condition)
# Experiment: use the p value as a continuous measure; compare the discriminating power of p>0.5 vs p<=0.5
print("Replace the holds binary with p-value buckets:")
for h, g in m.groupby("h"):
    high = g[g["p"] >= 0.5]
    low = g[g["p"] < 0.5]
    if len(high) > 0 and len(low) > 0:
        r1 = high[outcome].mean()
        r0 = low[outcome].mean()
        lift = r1 / r0 if r0 > 0 else float("nan")
        print(f"  h={h}: p>=0.5 outcome rate={r1:.4f} (n={len(high)}), p<0.5 outcome rate={r0:.4f} (n={len(low)}), lift ratio={lift:.2f}")

print()
print("Stricter buckets with p>=0.7:")
for h, g in m.groupby("h"):
    high = g[g["p"] >= 0.7]
    mid = g[(g["p"] >= 0.3) & (g["p"] < 0.7)]
    low = g[g["p"] < 0.3]
    for name, sub in [("high (p>=0.7)", high), ("mid (0.3-0.7)", mid), ("low (p<0.3)", low)]:
        if len(sub) > 0:
            print(f"  h={h} {name}: n={len(sub)}, outcome rate={sub[outcome].mean():.4f}")

# ---------- 4. Outcome label distribution ----------
print()
print("=" * 70)
print("4. Outcome label distribution")
print("=" * 70)
print(f"Overall above_median rate: {m[outcome].mean():.4f}")
for h, g in m.groupby("h"):
    print(f"  h={h}: above_median rate={g[outcome].mean():.4f}, n={len(g)}")

# ---------- 5. Firm features: holds=1 vs holds=0 ----------
print()
print("=" * 70)
print("5. Firm features: holds=1 vs holds=0")
print("=" * 70)
feat_vars = ["n_jobs", "mean_size", "n_mod", "e2_share", "struct_sim", "xfirm_sim", "surf_sim"]
for var in feat_vars:
    h1 = m[m["holds"] == 1][var]
    h0 = m[m["holds"] == 0][var]
    t, p = ttest(h1, h0)
    print(f"  {var}: holds=1 mean={h1.mean():.4f} (n={len(h1)}), "
          f"holds=0 mean={h0.mean():.4f} (n={len(h0)}), t={t:.3f}, p={p:.4f}")

# Firm features: split=1 vs split=0
print()
print("--- Firm features: above_median=1 vs 0 ---")
for var in feat_vars:
    sp = m[m[outcome] == 1][var]
    ns = m[m[outcome] == 0][var]
    t, p = ttest(sp, ns)
    print(f"  {var}: outcome=1 mean={sp.mean():.4f} (n={len(sp)}), "
          f"outcome=0 mean={ns.mean():.4f} (n={len(ns)}), t={t:.3f}, p={p:.4f}")

# ---------- 6. Logistic regression ----------
print()
print("=" * 70)
print("6. Logistic regression (effect of holds after controlling for firm characteristics)")
print("=" * 70)
feat_names = ["holds"] + feat_vars
X = m[feat_names].copy()
for c in feat_names:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
y = m[outcome].values
X = np.column_stack([np.ones(len(X)), X.values])

beta = np.zeros(X.shape[1])
for _ in range(200):
    eta = X @ beta
    p_pred = 1 / (1 + np.exp(-eta))
    W = np.clip(p_pred * (1 - p_pred), 1e-10, None)
    grad = X.T @ (y - p_pred)
    H = X.T @ (X * W[:, None])
    try:
        delta = np.linalg.solve(H + 1e-8 * np.eye(H.shape[0]), grad)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(H, grad, rcond=None)[0]
    beta = beta + delta
    if np.max(np.abs(delta)) < 1e-8:
        break

eta = X @ beta
p_pred = 1 / (1 + np.exp(-eta))
W = np.clip(p_pred * (1 - p_pred), 1e-10, None)
cov = np.linalg.inv(X.T @ (X * W[:, None]))
se = np.sqrt(np.diag(cov))

print("variable          coef      std err    z       p-value")
print("-" * 55)
for i, name in enumerate(["const"] + feat_names):
    z = beta[i] / se[i] if se[i] > 0 else float("nan")
    pval = math.erfc(abs(z) / math.sqrt(2))
    sign = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"{name:16s} {beta[i]:9.4f}  {se[i]:8.4f}  {z:8.3f}  {pval:.4f} {sign}")

or_holds = math.exp(beta[1])
print(f"\nholds OR = {or_holds:.4f}")
if abs(or_holds - 1) < 0.1:
    print("After controlling for firm characteristics, holds has almost no independent effect on above_median; vacuous confirmed.")
elif or_holds > 1:
    print("After controlling for firm characteristics, holds still has a positive effect, indicating it was previously masked by confounders.")
else:
    print("After controlling for firm characteristics, holds shows an inverse effect.")
