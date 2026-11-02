"""Merge LLM antecedent judgments with outcome labels, producing "antecedent → outcome" conditional probability tables.

Inputs: out/judge_output/{g1,g2,g3}_batch_*.json (LLM judgments)
        out/objects/labels_g{1,2,3}_2020.parquet (outcome labels)
Outputs: out/objects/judge_g{1,2,3}_2020.parquet (per-object judgments persisted)
         out/results/antecedent_outcome_2020.md (conditional probability summary tables)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent")
OUT = BASE / "out"
JD = OUT / "judge_output"
OBJ = OUT / "objects"
RES = OUT / "results"

RULES = {
    "g1": ("job", "mixed_new"),        # (a) job task set → mixed new combination
    "g2": ("seed", "split"),           # (b) task cluster → split
    "g3": ("firm", "above_median"),    # (c) firm → template adoption share above median
}


def load_judgments(rule: str) -> pd.DataFrame:
    files = sorted(JD.glob(f"{rule}_batch_*.json"))
    if not files:
        raise FileNotFoundError(f"{rule}: no judgment files found")
    rows = []
    for f in files:
        try:
            arr = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  [skip] {f.name} JSON parse failed: {e}")
            continue
        if not isinstance(arr, list):
            print(f"  [skip] {f.name} is not a JSON array")
            continue
        for x in arr:
            if not isinstance(x, dict) or "id" not in x:
                continue
            rows.append({
                rule: str(x["id"]),
                "holds": int(x.get("holds", 0) in (1, "1", True)),
                "p": float(x.get("p", 0.0)),
                "why": str(x.get("why", ""))[:60],
                "batch": f.stem,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=[rule], keep="first")
    return df


def cond_table(j: pd.DataFrame, lab: pd.DataFrame, key: str, outcome: str, rule: str) -> pd.DataFrame:
    m = j.merge(lab, on=key, how="inner")
    rows = []
    for h, g in m.groupby("h"):
        a = g[g["holds"] == 1]
        b = g[g["holds"] == 0]
        base = g[outcome].mean()
        p1 = a[outcome].mean() if len(a) else float("nan")
        p0 = b[outcome].mean() if len(b) else float("nan")
        rows.append({
            "rule": rule, "h": int(h), "n": len(g),
            "antecedent hold rate": round(g["holds"].mean(), 4),
            "P(outcome|holds=1)": round(p1, 4) if pd.notna(p1) else None,
            "P(outcome|holds=0)": round(p0, 4) if pd.notna(p0) else None,
            "base rate": round(base, 4),
            "lift ratio (holds=1/0)": round(p1 / p0, 2) if pd.notna(p1) and pd.notna(p0) and p0 > 0 else None,
            "n1": len(a), "n0": len(b),
        })
    return pd.DataFrame(rows)


def calib_table(j: pd.DataFrame, lab: pd.DataFrame, key: str, outcome: str) -> pd.DataFrame:
    m = j.merge(lab, on=key, how="inner")
    m["p bucket"] = m["p"].map({0.05: "0.05", 0.1: "0.1", 0.2: "0.2", 0.3: "0.3",
                                0.5: "0.5", 0.7: "0.7", 0.9: "0.9"}).fillna(m["p"].round(2).astype(str))
    return (m.groupby(["h", "p bucket"])[outcome]
            .agg(["mean", "count"]).rename(columns={"mean": "actual outcome rate", "count": "n"})
            .round(4).reset_index())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    RES.mkdir(parents=True, exist_ok=True)
    all_tables, all_calib = [], []
    for rule, (key, outcome) in RULES.items():
        j = load_judgments(rule).rename(columns={rule: key})
        pool = pd.read_parquet(OBJ / f"{rule}_{'jobs' if rule=='g1' else 'comm' if rule=='g2' else 'firms'}_{2020}.parquet")
        known = set(pool[key].astype(str))
        miss = len(j) - j[key].isin(known).sum()
        j.to_parquet(OBJ / f"judge_{rule}_2020.parquet", index=False)
        lab = pd.read_parquet(OBJ / f"labels_{rule}_2020.parquet")
        lab[key] = lab[key].astype(str)
        t = cond_table(j, lab, key, outcome, rule)
        all_tables.append(t)
        c = calib_table(j, lab, key, outcome)
        c.insert(0, "rule", rule)
        all_calib.append(c)
        n_merged = j.merge(lab, on=key, how="inner").shape[0]
        print(f"\n=== {rule} ({len(j)} objects judged, of which id not in pool: {miss}; {n_merged} after aligning judgments with labels) ===")
        print(t.to_string(index=False))

    tt = pd.concat(all_tables)
    cc = pd.concat(all_calib)
    lines = ["# Conditional probability table of the three criteria \"antecedent → years later → outcome\" (T=2020)", ""]
    lines.append("Judgment source: LLM per-object antecedent judgment (out/judge_output/); outcome: labels_g*_2020 (measured at T+h).")
    lines.append("The criterion is supported when P(outcome|holds=1) is significantly higher than P(outcome|holds=0) and the lift ratio is > 1.")
    lines.append("")
    lines.append("## Main conditional probability table")
    lines.append("")
    lines.append(tt.to_markdown(index=False))
    lines.append("")
    lines.append("## Probability bin calibration table (LLM p buckets vs actual outcome rate)")
    lines.append("")
    lines.append(cc.to_markdown(index=False))
    md = "\n".join(lines)
    (RES / "antecedent_outcome_2020.md").write_text(md, encoding="utf-8")
    print(f"\nWritten to {RES / 'antecedent_outcome_2020.md'}")


if __name__ == "__main__":
    main()
