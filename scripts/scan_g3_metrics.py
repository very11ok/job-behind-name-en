"""g3 fix: from LLM subjective judgment to objective structural metrics.

The diagnosis confirmed that the LLM judgment carries no information (p-bucket ordering power ≈ 0; holds has no independent effect after controlling for firm characteristics).
This script tests the predictive power of each objective structural metric as a binary antecedent, looking for a predictive replacement.

Metrics tested:
  - surf_sim: surface similarity (surface similarity to the template)
  - struct_sim: structural similarity
  - xfirm_sim: cross-firm similarity
  - n_jobs: number of firm jobs
  - n_mod: number of modules
  - e2_share: codable task share
  - mean_size: mean job size
  - residual: surf_sim - struct_sim (the basis of the original LLM judgment)
  - residual2: surf_sim - xfirm_sim
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent")
OUT = BASE / "out"
OBJ = OUT / "objects"
RES = OUT / "results"


def ttest_p(a_rate: float, a_n: int, b_rate: float, b_n: int) -> float:
    """p value of the two-proportion z test (normal approximation)."""
    if a_n < 5 or b_n < 5:
        return float("nan")
    p_pool = (a_rate * a_n + b_rate * b_n) / (a_n + b_n)
    if p_pool == 0 or p_pool == 1:
        return float("nan")
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / a_n + 1 / b_n))
    if se == 0:
        return float("nan")
    z = (a_rate - b_rate) / se
    return math.erfc(abs(z) / math.sqrt(2))


def cond_for_metric(m: pd.DataFrame, metric: str, label: str) -> list[dict]:
    """Binary-split a single metric (median split) and compute the conditional probabilities under each h."""
    med = m[metric].median()
    m = m.assign(_high=m[metric] >= med, _low=m[metric] < med)
    rows = []
    for h, g in m.groupby("h"):
        hi = g[g["_high"]]
        lo = g[g["_low"]]
        p1 = hi["above_median"].mean() if len(hi) else float("nan")
        p0 = lo["above_median"].mean() if len(lo) else float("nan")
        lift = p1 / p0 if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")
        p = ttest_p(p1, len(hi), p0, len(lo))
        rows.append({
            "metric": label, "h": int(h), "n": len(g),
            "n_high": len(hi), "n_low": len(lo),
            "P(adopt|high)": round(p1, 4) if pd.notna(p1) else None,
            "P(adopt|low)": round(p0, 4) if pd.notna(p0) else None,
            "lift ratio": round(lift, 2) if pd.notna(lift) else None,
            "p value": f"{p:.2e}" if pd.notna(p) else None,
        })
    return rows


def corr_for_metric(m: pd.DataFrame, metric: str, label: str) -> list[dict]:
    """Spearman correlation between the continuous variable and adopt_share."""
    rows = []
    for h, g in m.groupby("h"):
        corr = g[metric].corr(g["adopt_share"], method="spearman")
        rows.append({
            "metric": label, "h": int(h),
            "Spearman correlation": round(corr, 4) if pd.notna(corr) else None,
            "n": len(g),
        })
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # Merge data
    firms = pd.read_parquet(OBJ / "g3_firms_2020.parquet")
    j = pd.read_parquet(OBJ / "judge_g3_2020.parquet")
    lab = pd.read_parquet(OBJ / "labels_g3_2020.parquet")

    for df in [firms, j, lab]:
        df["firm"] = df["firm"].astype(str)

    m = j.merge(firms, on="firm", how="inner").merge(
        lab[["firm", "h", "above_median", "adopt_share"]], on="firm", how="inner"
    )

    # Compute residuals
    m["residual"] = m["surf_sim"] - m["struct_sim"]
    m["residual2"] = m["surf_sim"] - m["xfirm_sim"]

    print(f"Total sample: {len(m)} (holds=1: {len(m[m['holds']==1])}, holds=0: {len(m[m['holds']==0])})")
    print(f"Samples per h: {m.groupby('h').size().to_dict()}")
    print()

    # 1. LLM judgment baseline
    print("=== LLM judgment (baseline) ===")
    llm_rows = cond_for_metric(m.assign(_metric=m["holds"]), "holds", "LLM judgment")
    # Use holds as the binary split
    for h, g in m.groupby("h"):
        hi = g[g["holds"] == 1]
        lo = g[g["holds"] == 0]
        p1 = hi["above_median"].mean() if len(hi) else float("nan")
        p0 = lo["above_median"].mean() if len(lo) else float("nan")
        lift = p1 / p0 if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")
        p = ttest_p(p1, len(hi), p0, len(lo))
        print(f"  h={h}: P(adopt|holds=1)={p1:.4f}, P(adopt|holds=0)={p0:.4f}, lift ratio={lift:.2f}, p={p:.2e}")

    print()

    # 2. Objective metric scan (binary split)
    metrics = [
        ("surf_sim", "surface similarity"),
        ("struct_sim", "structural similarity"),
        ("xfirm_sim", "cross-firm similarity"),
        ("n_jobs", "number of jobs"),
        ("n_mod", "number of modules"),
        ("e2_share", "codable task share"),
        ("mean_size", "mean job size"),
        ("residual", "residual (surf-struct)"),
        ("residual2", "residual (surf-xfirm)"),
    ]

    all_cond = []
    all_corr = []

    for col, label in metrics:
        rows = cond_for_metric(m, col, label)
        all_cond.extend(rows)
        corr_rows = corr_for_metric(m, col, label)
        all_corr.extend(corr_rows)
        print(f"=== {label} ({col}) ===")
        for r in rows:
            print(f"  h={r['h']}: P(high)={r['P(adopt|high)']}, P(low)={r['P(adopt|low)']}, lift ratio={r['lift ratio']}, p={r['p value']}")
        print()

    # 3. Continuous correlations
    print("=== Continuous Spearman correlations ===")
    corr_df = pd.DataFrame(all_corr)
    print(corr_df.to_string(index=False))
    print()

    # Save
    RES.mkdir(parents=True, exist_ok=True)
    cond_df = pd.DataFrame(all_cond)
    cond_df.to_parquet(RES / "g3_metric_scan.parquet", index=False)
    corr_df.to_parquet(RES / "g3_metric_corr.parquet", index=False)

    # Write markdown
    lines = ["# g3 objective metric scan: from LLM judgment to structural metrics", ""]
    lines.append(f"Total sample: {len(m)}; n under each h = {m.groupby('h').size().to_dict()}")
    lines.append("")
    lines.append("## LLM judgment baseline")
    lines.append("")
    lines.append("| h | P(adopt|holds=1) | P(adopt|holds=0) | lift ratio | p value |")
    lines.append("|---|-------------------|-------------------|--------|------|")
    for h, g in m.groupby("h"):
        hi = g[g["holds"] == 1]
        lo = g[g["holds"] == 0]
        p1 = hi["above_median"].mean() if len(hi) else float("nan")
        p0 = lo["above_median"].mean() if len(lo) else float("nan")
        lift = p1 / p0 if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")
        p = ttest_p(p1, len(hi), p0, len(lo))
        lines.append(f"| {h} | {p1:.4f} | {p0:.4f} | {lift:.2f} | {p:.2e} |")
    lines.append("")
    lines.append("## Conditional probabilities of objective metrics (median split)")
    lines.append("")
    lines.append(cond_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Continuous Spearman correlations")
    lines.append("")
    lines.append(corr_df.to_markdown(index=False))
    lines.append("")

    md = "\n".join(lines)
    (RES / "g3_metric_scan.md").write_text(md, encoding="utf-8")
    print(f"Written {RES / 'g3_metric_scan.md'}")


if __name__ == "__main__":
    main()
