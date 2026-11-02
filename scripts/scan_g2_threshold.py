"""g2 threshold scan: search for size/hier_n cut points that make the inverse
relationship disappear or turn positive.

The diagnosis found that the g2 inverse relationship is driven by confounding
variables — the LLM assigned "the antecedent holds" to small simple clusters,
while the clusters that actually split are large complex ones. This script scans
threshold values of size and hier_n, computes conditional probabilities at each
threshold, and looks for the cut point where P(split|holds=1) >= P(split|holds=0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent")
OUT = BASE / "out"
OBJ = OUT / "objects"
RES = OUT / "results"


def merge_g2() -> pd.DataFrame:
    """Merge judgments, cluster features, and outcome labels."""
    j = pd.read_parquet(OBJ / "judge_g2_2020.parquet")
    comm = pd.read_parquet(OBJ / "g2_comm_2020.parquet")
    lab = pd.read_parquet(OBJ / "labels_g2_2020.parquet")

    j["seed"] = j["seed"].astype(str)
    comm["seed"] = comm["seed"].astype(str)
    lab["seed"] = lab["seed"].astype(str)

    m = j.merge(comm[["seed", "size", "n_mod", "e2_share", "hier_n", "mod"]],
                on="seed", how="inner")
    m = m.merge(lab[["seed", "split"]], on="seed", how="inner")
    return m


def cond_at_threshold(m: pd.DataFrame, col: str, threshold: int) -> dict:
    """Compute conditional probabilities at the given threshold."""
    sub = m[m[col] >= threshold]
    n = len(sub)
    if n < 20:
        return {"threshold": threshold, "n": n, "valid": False}

    h1 = sub[sub["holds"] == 1]
    h0 = sub[sub["holds"] == 0]
    p1 = h1["split"].mean() if len(h1) else float("nan")
    p0 = h0["split"].mean() if len(h0) else float("nan")
    lift = p1 / p0 if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")

    return {
        "threshold": threshold,
        "n": n,
        "n_h1": len(h1),
        "n_h0": len(h0),
        "P(split|holds=1)": round(p1, 4) if pd.notna(p1) else None,
        "P(split|holds=0)": round(p0, 4) if pd.notna(p0) else None,
        "lift ratio": round(lift, 2) if pd.notna(lift) else None,
        "valid": True,
    }


def scan_thresholds(m: pd.DataFrame) -> pd.DataFrame:
    """Scan multiple threshold values of size and hier_n."""
    rows = []

    # size threshold scan
    for th in [1, 3, 5, 7, 9, 11, 13, 15, 20, 25]:
        r = cond_at_threshold(m, "size", th)
        r["variable"] = "size"
        rows.append(r)

    # hier_n threshold scan
    for th in [1, 2, 3, 4, 5, 6]:
        r = cond_at_threshold(m, "hier_n", th)
        r["variable"] = "hier_n"
        rows.append(r)

    return pd.DataFrame(rows)


def scan_combined(m: pd.DataFrame) -> pd.DataFrame:
    """Scan joint thresholds of size and hier_n."""
    rows = []
    for s_th in [3, 5, 7, 9, 11]:
        for h_th in [2, 3, 4]:
            sub = m[(m["size"] >= s_th) & (m["hier_n"] >= h_th)]
            n = len(sub)
            if n < 20:
                continue
            h1 = sub[sub["holds"] == 1]
            h0 = sub[sub["holds"] == 0]
            p1 = h1["split"].mean() if len(h1) else float("nan")
            p0 = h0["split"].mean() if len(h0) else float("nan")
            lift = p1 / p0 if pd.notna(p1) and pd.notna(p0) and p0 > 0 else float("nan")
            rows.append({
                "size threshold": s_th,
                "hier_n threshold": h_th,
                "n": n,
                "n_h1": len(h1),
                "n_h0": len(h0),
                "P(split|holds=1)": round(p1, 4) if pd.notna(p1) else None,
                "P(split|holds=0)": round(p0, 4) if pd.notna(p0) else None,
                "lift ratio": round(lift, 2) if pd.notna(lift) else None,
            })
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m = merge_g2()

    print(f"Total sample: {len(m)}")
    print(f"holds=1: {len(m[m['holds']==1])}, holds=0: {len(m[m['holds']==0])}")
    print(f"split base rate: {m['split'].mean():.4f}")
    print()

    # Full-sample baseline
    full = {
        "P(split|holds=1)": round(m[m["holds"] == 1]["split"].mean(), 4),
        "P(split|holds=0)": round(m[m["holds"] == 0]["split"].mean(), 4),
    }
    lift_full = full["P(split|holds=1)"] / full["P(split|holds=0)"]
    full["lift ratio"] = round(lift_full, 2)
    print(f"Full-sample baseline: {full}")
    print()

    # Univariate threshold scan
    scan = scan_thresholds(m)
    print("=== Univariate threshold scan ===")
    print(scan.to_string(index=False))
    print()

    # Joint threshold scan
    combo = scan_combined(m)
    print("=== Joint threshold scan ===")
    print(combo.to_string(index=False))
    print()

    # Feature distribution: holds=1 vs holds=0
    print("=== Task-cluster characteristics by holds group ===")
    for var in ["size", "n_mod", "e2_share", "hier_n"]:
        h1 = m[m["holds"] == 1][var]
        h0 = m[m["holds"] == 0][var]
        print(f"{var:12s}  holds=1: mean={h1.mean():.2f}  median={h1.median():.0f}  "
              f"|  holds=0: mean={h0.mean():.2f}  median={h0.median():.0f}")

    # Save results
    RES.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(RES / "g2_threshold_scan.parquet", index=False)
    combo.to_parquet(RES / "g2_threshold_combo.parquet", index=False)

    # Write markdown report
    lines = ["# g2 threshold scan: searching for the optimal cut point", ""]
    lines.append(f"Total sample: {len(m)} (holds=1: {len(m[m['holds']==1])}, holds=0: {len(m[m['holds']==0])})")
    lines.append(f"split base rate: {m['split'].mean():.4f}")
    lines.append("")
    lines.append(f"**Full-sample baseline**: P(split|holds=1)={full['P(split|holds=1)']}, "
                 f"P(split|holds=0)={full['P(split|holds=0)']}, lift ratio={full['lift ratio']}")
    lines.append("")
    lines.append("## Univariate threshold scan")
    lines.append("")
    lines.append(scan.to_markdown(index=False))
    lines.append("")
    lines.append("## Joint threshold scan")
    lines.append("")
    lines.append(combo.to_markdown(index=False))
    lines.append("")
    lines.append("## Task-cluster characteristics by holds group")
    lines.append("")
    lines.append("| variable | holds=1 (mean/median) | holds=0 (mean/median) |")
    lines.append("|------|----------------------|----------------------|")
    for var in ["size", "n_mod", "e2_share", "hier_n"]:
        h1 = m[m["holds"] == 1][var]
        h0 = m[m["holds"] == 0][var]
        lines.append(f"| {var} | {h1.mean():.2f} / {h1.median():.0f} | {h0.mean():.2f} / {h0.median():.0f} |")

    md = "\n".join(lines)
    (RES / "g2_threshold_scan.md").write_text(md, encoding="utf-8")
    print(f"\nWritten {RES / 'g2_threshold_scan.md'}")


if __name__ == "__main__":
    main()
