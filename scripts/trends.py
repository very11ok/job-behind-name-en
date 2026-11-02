"""Signature trends of criteria (a)/(b): task count of new bundles, E variance, high/low exposure mix.

(b) Hierarchical unbundling → new jobs narrower and more homogeneous (size↓, E-var↓)
(a) Complementarity re-bundling → more co-bundles mixing high and low exposure (mixed share↑)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from jtagent import pairs as P

OUT = Path(r"D:\harness office\research\jobtask-agent\out")


def run():
    fa = P.first_appear().rename(columns={"year": "first_year"})
    td = pd.read_parquet(OUT / "snapshots" / "task_dict.parquet")[["task", "E_ord"]]
    e_map = dict(zip(td["task"], td["E_ord"]))

    rows = []
    for y in range(2016, 2027):
        jobs = P.load_jobs(y)
        jobs = jobs.merge(fa, on="job", how="left")
        new = jobs[jobs["first_year"] == y]
        stats = {"size": [], "e_var": [], "emean": [], "mixed": [], "contains_E2": []}
        for ts in new["tasks"]:
            es = np.array([e_map.get(int(t), 1) for t in ts], dtype=float)
            stats["size"].append(len(es))
            if len(es) >= 2:
                stats["e_var"].append(float(es.var()))
                stats["emean"].append(float(es.mean()))
                stats["mixed"].append(float(es.min() == 0 and es.max() == 2))
                stats["contains_E2"].append(float(es.max() == 2))
        size = np.array(stats["size"])
        rows.append({
            "year": y,
            "n_new": len(new),
            "size_mean": round(float(size.mean()), 3),
            "size_p25": int(np.percentile(size, 25)) if len(size) else 0,
            "size_p50": int(np.percentile(size, 50)) if len(size) else 0,
            "e_var_mean": round(float(np.mean(stats["e_var"])), 4),
            "mixed_E0xE2_share": round(float(np.mean(stats["mixed"])), 4),
            "contains_E2_share": round(float(np.mean(stats["contains_E2"])), 4),
        })
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "results" / "signature_trend.csv", index=False, encoding="utf-8-sig")
    print(tr.to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
