"""Inverse probability table: starting from the outcome, the share of antecedent-holds among objects where the outcome occurred.

Bayesian inverse of the forward table in merge_judge.py:
  forward  P(outcome | holds=1)  vs  P(outcome | holds=0)
  inverse  P(holds=1 | outcome=1)  vs  P(holds=1 | outcome=0)

Inverse lift ratio = P(holds=1|outcome=1) / P(holds=1|outcome=0)
  >1: when the outcome occurs, the antecedent is more likely to hold (the criterion has explanatory power)
  <1: when the outcome occurs, the antecedent is less likely to hold (inverse relation)

Inputs: out/objects/judge_g{1,2,3}_2020.parquet (LLM judgments, persisted)
        out/objects/labels_g{1,2,3}_2020.parquet (outcome labels)
Output: out/results/reverse_outcome_antecedent_2020.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent")
OUT = BASE / "out"
OBJ = OUT / "objects"
RES = OUT / "results"

RULES = {
    "g1": ("job", "mixed_new"),        # (a) job task set → mixed new combination
    "g2": ("seed", "split"),           # (b) task cluster → split
    "g3": ("firm", "above_median"),    # (c) firm → template adoption share above median
}


def reverse_table(j: pd.DataFrame, lab: pd.DataFrame, key: str, outcome: str, rule: str) -> pd.DataFrame:
    """Group by h and compute the conditional probability of holds conditional on the outcome."""
    m = j.merge(lab, on=key, how="inner")
    rows = []
    for h, g in m.groupby("h"):
        a = g[g[outcome] == 1]    # outcome occurred
        b = g[g[outcome] == 0]    # outcome did not occur
        p1 = a["holds"].mean() if len(a) else float("nan")  # P(holds=1 | outcome=1)
        p0 = b["holds"].mean() if len(b) else float("nan")  # P(holds=1 | outcome=0)
        base = g["holds"].mean()                              # base hold rate
        rows.append({
            "rule": rule, "h": int(h), "n": len(g),
            "P(holds=1|outcome=1)": round(p1, 4) if pd.notna(p1) else None,
            "P(holds=1|outcome=0)": round(p0, 4) if pd.notna(p0) else None,
            "base hold rate": round(base, 4),
            "inverse lift ratio": round(p1 / p0, 2) if pd.notna(p1) and pd.notna(p0) and p0 > 0 else None,
            "n_outcome1": len(a), "n_outcome0": len(b),
        })
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    RES.mkdir(parents=True, exist_ok=True)
    all_tables = []
    for rule, (key, outcome) in RULES.items():
        j_path = OBJ / f"judge_{rule}_2020.parquet"
        if not j_path.exists():
            print(f"[error] {j_path} does not exist; run merge_judge.py first")
            sys.exit(1)
        j = pd.read_parquet(j_path)
        j[key] = j[key].astype(str)
        lab = pd.read_parquet(OBJ / f"labels_{rule}_2020.parquet")
        lab[key] = lab[key].astype(str)
        t = reverse_table(j, lab, key, outcome, rule)
        all_tables.append(t)
        print(f"\n=== {rule} inverse table ===")
        print(t.to_string(index=False))

    tt = pd.concat(all_tables)
    lines = [
        "# Inverse probability table of the three criteria \"outcome → antecedent\" (T=2020)",
        "",
        "The forward and inverse tables are Bayesian inverses of each other: the forward table looks at P(outcome|holds), the inverse at P(holds|outcome).",
        "Inverse lift ratio = P(holds=1|outcome=1) / P(holds=1|outcome=0).",
        "",
        "Interpretation:",
        "  >1 and far above the base rate → the antecedent holds more often when the outcome occurs; the criterion has explanatory power",
        "  ≈1 and close to the base rate → the antecedent and the outcome are essentially independent; the criterion has no discriminating power",
        "  <1 → the antecedent is less likely to hold when the outcome occurs; an inverse relationship",
        "",
        "When the base rate (P(holds=1)) is high, be wary: even if the criterion has no explanatory power, P(holds=1|outcome=1) will be pulled up.",
        "What is truly informative is the \"inverse lift ratio\" and the \"net correspondence after deducting the base rate\".",
        "",
        "## Inverse probability table",
        "",
        tt.to_markdown(index=False),
        "",
        "## Forward table comparison (lift ratio)",
        "",
        "| rule | h | forward lift ratio | inverse lift ratio | base hold rate |",
        "|:-----|---:|----------:|----------:|----------:|",
    ]
    # append the forward lift ratio comparison
    forward = pd.read_csv(RES / "antecedent_outcome_2020.md", sep="|", skiprows=7, engine="python")
    # simple approach: extracting directly from the forward md text is unreliable; recompute instead
    for rule, (key, outcome) in RULES.items():
        j = pd.read_parquet(OBJ / f"judge_{rule}_2020.parquet")
        j[key] = j[key].astype(str)
        lab = pd.read_parquet(OBJ / f"labels_{rule}_2020.parquet")
        lab[key] = lab[key].astype(str)
        m = j.merge(lab, on=key, how="inner")
        for h, g in m.groupby("h"):
            a = g[g["holds"] == 1]
            b = g[g["holds"] == 0]
            p1 = a[outcome].mean() if len(a) else float("nan")
            p0 = b[outcome].mean() if len(b) else float("nan")
            fwd = round(p1 / p0, 2) if pd.notna(p1) and pd.notna(p0) and p0 > 0 else None
            rev = tt[(tt["rule"] == rule) & (tt["h"] == int(h))]["inverse lift ratio"].iloc[0]
            base_rate = round(g["holds"].mean(), 4)
            lines.append(f"| {rule} | {int(h)} | {fwd} | {rev} | {base_rate} |")

    lines.append("")
    md = "\n".join(lines)
    out_path = RES / "reverse_outcome_antecedent_2020.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
