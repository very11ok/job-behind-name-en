"""Criterion 3 (mimetic template diffusion) fix verification: objective antecedents n_jobs / n_mod.

The LLM judgment is confirmed vacuous; switch to firm size-structure metrics:
  antecedent = n_jobs high (median split), n_mod high, and the combination of both
  outcome = above_median (among the firm's new jobs at T+h, the cross-firm template adoption share is above the same-year median)

Output: conditional probabilities + lift ratio + inverse probabilities + chi-square, compared against LLM holds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

OUT = Path(r"D:\harness office\research\jobtask-agent\out")
OBJ = OUT / "objects"
RES = OUT / "results"
YEAR = 2020


def z2(a, na, b, nb):
    if na == 0 or nb == 0:
        return float("nan")
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    if se == 0:
        return float("nan")
    return math.erfc(abs((p1 - p2) / se) / math.sqrt(2))


def chi2_2x2(a, b, c, d):
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return float("nan"), float("nan")
    x2 = (a * d - b * c) ** 2 * n / ((a + b) * (c + d) * (a + c) * (b + d))
    return x2, math.erfc(math.sqrt(x2 / 2))


def main():
    firms = pd.read_parquet(OBJ / f"g3_firms_{YEAR}.parquet")
    labels = pd.read_parquet(OBJ / f"labels_g3_{YEAR}.parquet")
    judge = pd.read_parquet(OBJ / f"judge_g3_{YEAR}.parquet")
    judge["firm"] = judge["firm"].astype(str)
    holds = dict(zip(judge["firm"], judge["holds"]))

    firms["firm"] = firms["firm"].astype(str)
    med_jobs = firms["n_jobs"].median()
    med_mod = firms["n_mod"].median()
    firms["jobs_high"] = (firms["n_jobs"] > med_jobs).astype(int)
    firms["mod_high"] = (firms["n_mod"] > med_mod).astype(int)
    firms["combo"] = ((firms["jobs_high"] == 1) & (firms["mod_high"] == 1)).astype(int)
    firms["holds"] = firms["firm"].map(holds).fillna(0).astype(int)

    m = labels.merge(firms[["firm", "jobs_high", "mod_high", "combo", "holds",
                            "n_jobs", "n_mod", "e2_share", "mean_size",
                            "struct_sim", "surf_sim"]],
                     on="firm", how="inner")
    print(f"Sample size after merge: {len(m)} (per h: {m.groupby('h').size().to_dict()})")
    print(f"n_jobs median={med_jobs}, n_mod median={med_mod}")

    ANTE = [
        ("n_jobs high", "jobs_high"),
        ("n_mod high", "mod_high"),
        ("n_jobs high ∧ n_mod high", "combo"),
        ("LLM holds (control)", "holds"),
    ]
    CONS = [("above_median", "template adoption above median"), ("adopt_share", None)]

    lines = ["# Criterion 3 fix verification: objective size antecedent → template adoption outcome", ""]
    lines.append(f"Sample: {len(m)} firm×h; n_jobs median={med_jobs}, n_mod median={med_mod}")
    lines.append("")

    for label, col in ANTE:
        lines.append(f"## Antecedent = {col}")
        for h in (1, 2, 3):
            g = m[m["h"] == h]
            a, b = g[g[col] == 1], g[g[col] == 0]
            pa, pb = a["above_median"].mean(), b["above_median"].mean()
            pv = z2(int(a["above_median"].sum()), len(a),
                    int(b["above_median"].sum()), len(b))
            ratio = pa / pb if pb > 0 else float("inf")
            # Inverse probability
            rev_a = a["above_median"].sum() / max(a["above_median"].sum()
                                                  + b["above_median"].sum(), 1)
            lines.append(f"  h={h}: P(high adoption|antecedent=1)={pa:.4f}(n={len(a)}), "
                         f"P(high adoption|antecedent=0)={pb:.4f}(n={len(b)}), "
                         f"lift ratio={ratio:.2f}, p={pv:.3g}")
            # adopt_share continuous mean comparison
            lines.append(f"        adopt_share mean: {a['adopt_share'].mean():.4f} vs "
                         f"{b['adopt_share'].mean():.4f}")
        lines.append("")

    # Chi-square
    lines.append("## Chi-square (antecedent × above_median)")
    for label, col in ANTE:
        row = [f"  {col}:"]
        for h in (1, 2, 3):
            g = m[m["h"] == h]
            a = int(((g[col] == 1) & (g["above_median"] == 1)).sum())
            b = int(((g[col] == 1) & (g["above_median"] == 0)).sum())
            c = int(((g[col] == 0) & (g["above_median"] == 1)).sum())
            d = int(((g[col] == 0) & (g["above_median"] == 0)).sum())
            x2, pv = chi2_2x2(a, b, c, d)
            row.append(f" h={h} chi2={x2:.2f} p={pv:.3g}")
        lines.append(" ".join(row))
    lines.append("")

    # Inverse probability: P(antecedent|outcome=1) vs P(antecedent|outcome=0)
    lines.append("## Inverse probability P(antecedent=1 | outcome)")
    for label, col in ANTE:
        for h in (1, 2, 3):
            g = m[m["h"] == h]
            p1 = g[g["above_median"] == 1][col].mean()
            p0 = g[g["above_median"] == 0][col].mean()
            lines.append(f"  h={h} {col}: P(antecedent|high adoption)={p1:.4f} vs "
                         f"P(antecedent|low adoption)={p0:.4f}, inverse lift ratio={p1 / p0 if p0 > 0 else float('inf'):.2f}")
    lines.append("")

    # Robustness: continuous Spearman
    lines.append("## Continuous Spearman (metrics vs adopt_share, per h)")
    for var in ("n_jobs", "n_mod", "e2_share", "mean_size", "struct_sim", "surf_sim"):
        for h in (1, 2, 3):
            g = m[(m["h"] == h) & m["adopt_share"].notna()]
            rho = g[var].rank().corr(g["adopt_share"].rank())
            lines.append(f"  h={h} {var}: rho={rho:.4f} (n={len(g)})")

    txt = "\n".join(lines)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "g3_fix_verify.md").write_text(txt, encoding="utf-8")
    m.to_parquet(RES / "g3_fix_verify.parquet", index=False)
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
