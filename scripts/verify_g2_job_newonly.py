"""Robustness check: is the job-version down_std driven by newly entering jobs?

holders are restricted to new jobs whose job id at year T+h is not in the T-year snapshot, and split / down_std are recomputed.
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
    if na == 0 or nb == 0:
        return float("nan")
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    if se == 0:
        return float("nan")
    return math.erfc(abs((p1 - p2) / se) / math.sqrt(2))


def main():
    obj = pd.read_parquet(OUT / "objects" / "g1_jobs_2020.parquet")
    attrs = pd.read_parquet(OUT / "snapshots" / "task_dict.parquet")
    e_map = dict(zip(attrs["task"], attrs["E_ord"].fillna(1)))
    jobs_now_ids = set(pd.read_parquet(OUT / "snapshots" / f"jobs_{YEAR}.parquet")["job"])

    jobs = {}
    for _, r in obj.iterrows():
        ts = _taskset(r["tasks"])
        es = [e_map.get(t, 1.0) for t in ts]
        jobs[r["job"]] = {"ts": ts, "e_std": float(np.std(es)), "n": len(ts)}
    med = float(np.median([j["e_std"] for j in jobs.values()]))

    rows = []
    for h in (1, 2, 3):
        fut = P.load_jobs(YEAR + h)
        # New job = job id not in the T-year snapshot; carrying by pre-existing jobs does not count as congealing into new jobs
        fut_new = [(frozenset(int(t) for t in r["tasks"]))
                   for _, r in fut.iterrows() if r["job"] not in jobs_now_ids]
        inv = {}
        for i, xj in enumerate(fut_new):
            for t in xj:
                inv.setdefault(t, []).append(i)
        n_new = len(fut_new)
        for jid, j in jobs.items():
            ts = j["ts"]
            cand = {i for t in ts for i in inv.get(t, [])}
            holders = [fut_new[i] for i in cand
                       if len(fut_new[i] & ts) >= 2 and len(fut_new[i]) < len(ts)]
            cover = (len(set().union(*[xj & ts for xj in holders])) / len(ts)
                     if holders else 0.0)
            split = int(len(holders) >= 2 and cover >= 0.6)
            h_std = [float(np.std([e_map.get(t, 1.0) for t in xj])) for xj in holders]
            down_std = int(bool(split) and bool(h_std) and np.mean(h_std) < j["e_std"])
            rows.append({"job": jid, "h": h, "split_new": split, "down_std_new": down_std,
                         "asym_high": int(j["e_std"] > med), "size": j["n"]})

    m = pd.DataFrame(rows)
    RES = OUT / "results"
    m.to_parquet(RES / "g2_job_newonly.parquet", index=False)
    print(f"Number of new jobs in year T+h: h=1/{sum(1 for _ in P.load_jobs(YEAR+1).itertuples())} total jobs, "
          f"of which new n_new={n_new}")
    print(f"asym median: {med:.4f}")
    print()
    for col in ("down_std_new", "split_new"):
        for h in (1, 2, 3):
            g = m[m["h"] == h]
            a, b = g[g["asym_high"] == 1], g[g["asym_high"] == 0]
            pa, pb = a[col].mean(), b[col].mean()
            pv = z2(int(a[col].sum()), len(a), int(b[col].sum()), len(b))
            print(f"{col} h={h}: P|asym high={pa:.4f}(n={len(a)}) vs P|asym low={pb:.4f}(n={len(b)}), "
                  f"lift ratio={pa / pb if pb > 0 else float('inf'):.2f}, p={pv:.3g}")
    print()
    print("Control (all jobs incl. pre-existing ones, from g2_job_verify): down_std 2.65/2.86/2.95")
    m.to_parquet(RES / "g2_job_newonly.parquet", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
