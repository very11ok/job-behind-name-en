"""Criterion 2 (hierarchical unbundling) directional fix verification.

New antecedent: the knowledge asymmetry of the task cluster (dispersion of E_ord across the tasks in the cluster).
New outcome: the asymmetry of the narrow jobs' task sets split off is lower than that of the original cluster (knowledge–routine vertical separation).

  asym(seed)  = std(E_ord of seed tasks) (also try range)
  directional = split==1 and the holders' mean asym < seed asym (downward unbundling)

Test: P(directional | asym high) vs P(directional | asym low), plus inverse probabilities and chi-square.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from jtagent import pairs as P

OUT = Path(r"D:\harness office\research\jobtask-agent\out")
YEAR = 2020


def _taskset(s) -> frozenset:
    return frozenset(int(x) for x in str(s).split(";") if str(x).strip())


def z2(a, na, b, nb):
    """Two-proportion z test; returns the p value."""
    if na == 0 or nb == 0:
        return float("nan")
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def chi2_2x2(a, b, c, d):
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return float("nan"), float("nan")
    x2 = (a * d - b * c) ** 2 * n / ((a + b) * (c + d) * (a + c) * (b + d))
    p = math.erfc(math.sqrt(x2 / 2))  # df=1 approximation
    return x2, p


def main():
    obj = pd.read_parquet(OUT / "objects" / "g2_comm_2020.parquet")
    attrs = pd.read_parquet(OUT / "snapshots" / "task_dict.parquet")
    e_map = dict(zip(attrs["task"], attrs["E_ord"].fillna(1)))
    judge = pd.read_parquet(OUT / "objects" / "judge_g2_2020.parquet")
    judge["seed"] = judge["seed"].astype(int)
    holds = dict(zip(judge["seed"], judge["holds"]))

    seeds = {}
    for _, r in obj.iterrows():
        ts = _taskset(r["tasks"])
        es = [e_map.get(t, 1.0) for t in ts]
        seeds[int(r["seed"])] = {
            "ts": ts, "n": len(ts),
            "e_mean": float(np.mean(es)),
            "e_std": float(np.std(es)),
            "e_range": float(max(es) - min(es)),
            "hier_n": int(r["hier_n"]),
            "holds": holds.get(int(r["seed"]), 0),
        }

    med_std = float(np.median([s["e_std"] for s in seeds.values()]))
    print(f"Number of seed clusters: {len(seeds)}, e_std median = {med_std:.4f}")

    rows = []
    for h in (1, 2, 3):
        f = OUT / "snapshots" / f"jobs_{YEAR + h}.parquet"
        if not f.exists():
            continue
        fut = P.load_jobs(YEAR + h)
        fut_jobs = [frozenset(int(t) for t in x) for x in fut["tasks"]]
        for sid, s in seeds.items():
            ts = s["ts"]
            holders = [xj for xj in fut_jobs
                       if len(xj & ts) >= 2 and len(xj) < len(ts)]
            cover = len(set().union(*[xj & ts for xj in holders])) / len(ts) if holders else 0.0
            split = int(len(holders) >= 2 and cover >= 0.6)
            # Directionality: the holders' mean task-set asymmetry is lower than the seed cluster's
            holder_std = [float(np.std([e_map.get(t, 1.0) for t in xj])) for xj in holders]
            holder_e = [float(np.mean([e_map.get(t, 1.0) for t in xj])) for xj in holders]
            down_std = int(bool(holder_std) and np.mean(holder_std) < s["e_std"])
            down_e = int(bool(holder_e) and np.mean(holder_e) < s["e_mean"])  # routine tasks sinking down to lower-skill work
            rows.append({
                "seed": sid, "h": h, "split": split,
                "down_std": int(split and down_std),
                "down_e": int(split and down_e),
                "asym_high": int(s["e_std"] > med_std),
                "e_std": s["e_std"], "e_range": s["e_range"],
                "hier_n": s["hier_n"], "holds": s["holds"],
            })

    m = pd.DataFrame(rows)
    RES = OUT / "results"
    RES.mkdir(parents=True, exist_ok=True)
    m.to_parquet(RES / "g2_direction_verify.parquet", index=False)

    lines = ["# Criterion 2 directional fix verification (asymmetry antecedent → downward unbundling outcome)", ""]
    lines.append(f"Seed clusters: {len(seeds)}; asym antecedent = e_std > median {med_std:.4f}")
    lines.append("")

    def blk(name, col):
        lines.append(f"## Antecedent = {name}")
        for h in (1, 2, 3):
            g = m[m["h"] == h]
            a = g[g["asym_high"] == 1]
            b = g[g["asym_high"] == 0]
            pa = a[col].mean() if len(a) else float("nan")
            pb = b[col].mean() if len(b) else float("nan")
            pv = z2(int(a[col].sum()), len(a), int(b[col].sum()), len(b))
            ratio = pa / pb if pb and pb == pb and pb > 0 else float("nan")
            lines.append(f"  h={h}: P({col}|asym high)={pa:.4f}(n={len(a)}), "
                         f"P({col}|asym low)={pb:.4f}(n={len(b)}), lift ratio={ratio:.2f}, p={pv:.3g}")
        lines.append("")

    blk("down_std (the split-off narrow jobs have lower asymmetry)", "down_std")
    blk("down_e (the split-off narrow jobs have lower mean E, routine tasks sinking down to lower-skill work)", "down_e")
    blk("split (original criterion, control)", "split")

    # Cross of LLM holds and asym
    lines.append("## Cross-tabulation: LLM holds × asym_high → split")
    for h in (1, 2, 3):
        g = m[m["h"] == h]
        for hv in (1, 0):
            for av in (1, 0):
                gg = g[(g["holds"] == hv) & (g["asym_high"] == av)]
                if len(gg):
                    lines.append(f"  h={h} holds={hv} asym_high={av}: "
                                 f"n={len(gg)}, split rate={gg['split'].mean():.4f}, "
                                 f"down_std rate={gg['down_std'].mean():.4f}")
        lines.append("")

    # Chi-square: asym_high → down_std
    lines.append("## Chi-square (asym_high × down_std)")
    for h in (1, 2, 3):
        g = m[m["h"] == h]
        a = int(((g["asym_high"] == 1) & (g["down_std"] == 1)).sum())
        b = int(((g["asym_high"] == 1) & (g["down_std"] == 0)).sum())
        c = int(((g["asym_high"] == 0) & (g["down_std"] == 1)).sum())
        d = int(((g["asym_high"] == 0) & (g["down_std"] == 0)).sum())
        x2, pv = chi2_2x2(a, b, c, d)
        lines.append(f"  h={h}: chi2={x2:.2f}, p={pv:.3g}  (2x2: {a},{b},{c},{d})")
    lines.append("")

    # Directionality under the holds=1 ∧ hier_n>=3 threshold (ties into the g2 fix conclusion)
    lines.append("## Combined antecedent: holds=1 ∧ hier_n>=3 ∧ asym_high → down_std")
    m["combo"] = ((m["holds"] == 1) & (m["hier_n"] >= 3) & (m["asym_high"] == 1)).astype(int)
    for h in (1, 2, 3):
        g = m[m["h"] == h]
        a, b = g[g["combo"] == 1], g[g["combo"] == 0]
        pa, pb = a["down_std"].mean(), b["down_std"].mean()
        pv = z2(int(a["down_std"].sum()), len(a), int(b["down_std"].sum()), len(b))
        lines.append(f"  h={h}: P={pa:.4f}(n={len(a)}) vs {pb:.4f}(n={len(b)}), "
                     f"lift ratio={pa / pb if pb > 0 else float('inf'):.2f}, p={pv:.3g}")

    txt = "\n".join(lines)
    (RES / "g2_direction_verify.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
