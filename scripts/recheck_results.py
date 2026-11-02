"""Independent recheck: recompute the key tests of the three-criteria fixes with
scipy.stats and compare against the hand-written formulas."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(r"D:\harness office\research\jobtask-agent\out")
RES = OUT / "results"

print("=== 1) Criterion 3 g3_fix_verify recheck ===")
m = pd.read_parquet(RES / "g3_fix_verify.parquet")
for col in ("jobs_high", "mod_high", "combo", "holds"):
    for h in (1, 2, 3):
        g = m[m["h"] == h]
        a = int(((g[col] == 1) & (g["above_median"] == 1)).sum())
        b = int(((g[col] == 1) & (g["above_median"] == 0)).sum())
        c = int(((g[col] == 0) & (g["above_median"] == 1)).sum())
        d = int(((g[col] == 0) & (g["above_median"] == 0)).sum())
        ct = np.array([[a, b], [c, d]])
        x2, p, _, _ = stats.chi2_contingency(ct, correction=False)
        pa, pb = a / (a + b), c / (c + d)
        print(f"  {col} h={h}: P1={pa:.4f} P0={pb:.4f} lift={pa / pb:.2f} "
              f"chi2={x2:.2f} p={p:.3g}")

print()
print("=== 2) Criterion 2 g2_direction_verify recheck ===")
d2 = pd.read_parquet(RES / "g2_direction_verify.parquet")
for h in (1, 2, 3):
    g = d2[d2["h"] == h]
    a = int(((g["asym_high"] == 1) & (g["down_std"] == 1)).sum())
    b = int(((g["asym_high"] == 1) & (g["down_std"] == 0)).sum())
    c = int(((g["asym_high"] == 0) & (g["down_std"] == 1)).sum())
    dd = int(((g["asym_high"] == 0) & (g["down_std"] == 0)).sum())
    ct = np.array([[a, b], [c, dd]])
    x2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    pa, pb = a / (a + b), c / (c + dd)
    print(f"  asym→down_std h={h}: P1={pa:.4f} P0={pb:.4f} lift={pa / pb:.2f} "
          f"chi2={x2:.2f} p={p:.3g}")

print()
print("=== 3) Criterion 2 threshold version g2_fix_verify recheck ===")
d3 = pd.read_parquet(RES / "g2_fix_verify.parquet")
print("  columns:", list(d3.columns))

print()
print("=== 4) Sample consistency check ===")
obj2 = pd.read_parquet(OUT / "objects" / "g2_comm_2020.parquet")
lab2 = pd.read_parquet(OUT / "objects" / "labels_g2_2020.parquet")
print(f"  g2 seed clusters: {obj2['seed'].nunique()} clusters / label rows {len(lab2)} "
      f"(= {obj2['seed'].nunique()} × 3 = {obj2['seed'].nunique() * 3})")
obj3 = pd.read_parquet(OUT / "objects" / "g3_firms_2020.parquet")
lab3 = pd.read_parquet(OUT / "objects" / "labels_g3_2020.parquet")
print(f"  g3 firms: {obj3['firm'].nunique()} firms / label rows {len(lab3)} (firms with no new jobs are missing rows for some h)")
judge2 = pd.read_parquet(OUT / "objects" / "judge_g2_2020.parquet")
judge3 = pd.read_parquet(OUT / "objects" / "judge_g3_2020.parquet")
print(f"  judge g2: {len(judge2)} rows (seed covers {judge2['seed'].astype(int).nunique()})"
      f" / judge g3: {len(judge3)} rows")

print()
print("=== 5) Missing and NaN check ===")
print(f"  above_median NaN in g3_fix m: {m['above_median'].isna().sum()}")
print(f"  g3 adopt_share NaN rows: {m['adopt_share'].isna().sum()} (firms with n_new_jobs=0)")
print(f"  rows where missing holds was filled with 0: {(m['holds'] == 0).sum()}")
