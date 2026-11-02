"""Object pool construction for the three criteria: defining units, candidate objects, and sampling.

Criterion 1 (a) unit = job task set; antecedent = high-exposure codable tasks × context-embedded tasks co-occurring.
Criterion 2 (b) unit = task cluster (connected cluster in the projection layer); antecedent = codable within the same hierarchy, neighborhood cohesion.
Criterion 3 (c) unit = firm × year; antecedent = cross-firm template residue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from jtagent import pairs as P
from jtagent.attrs import load_attrs

OUT = Path(r"D:\harness office\research\jobtask-agent\out")
OBJ = OUT / "objects"


def _attrs():
    return load_attrs(mode="llm")


def g1_jobs(attrs: pd.DataFrame, year: int, n_sample: int = 2000, seed=0) -> pd.DataFrame:
    """Criterion 1 object pool: all job task sets of that year (≥2 tasks), with the fields needed for a first-pass antecedent judgment."""
    jobs = P.load_jobs(year)
    a = attrs.set_index("task")
    recs = []
    for _, r in jobs.iterrows():
        ts = [t for t in r["tasks"] if t in a.index]
        if len(ts) < 2:
            continue
        sub = a.loc[ts]
        recs.append({
            "job": r["job"], "year": year, "size": len(ts),
            "tasks": ";".join(str(int(t)) for t in ts),
            "e_ord": ",".join(f"{v:.0f}" for v in sub["E_ord"].fillna(1)),
            "cod": ",".join(f"{v:.2f}" if pd.notna(v) else "na" for v in sub["codable"]),
            "ctx": ",".join(f"{v:.2f}" if pd.notna(v) else "na" for v in sub["ctx"]),
            "mod": "|".join(str(m) for m in sub["module"]),
            "has_E2": int((sub["E_ord"] >= 2).any()),
            "has_E0": int((sub["E_ord"] <= 0).any()),
            "n_mod": sub["module"].nunique(),
        })
    df = pd.DataFrame(recs)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed).reset_index(drop=True)
    return df


def g2_communities(year: int, n_sample: int = 2000, seed=0) -> pd.DataFrame:
    """Criterion 2 object pool: task cluster = a tight task cluster in the projection layer made of one task and its strong neighborhood.

    Taking each task as the seed, its neighbors are the core tasks with the highest weights in the weighted
    projection that belong to the same community, forming clusters of 3–15 tasks. This yields enough testable
    objects, and the "cluster" is indeed a set of tasks connected on the projection layer.
    """
    cdir = Path(r"D:\wanted\exposure\new material\community") / str(year)
    core = pd.read_csv(cdir / f"core_tasks_{year}.csv")
    core = core[core["is_core"] == True]
    comm_of = dict(zip(core["task_id"].astype(int), core["comm"].astype(int)))

    edges = P.load_edges(year)
    tasks, tidx, A, deg_job, deg_pair = P.build_projection(edges)
    W = A.tocsr()  # 0/1 co-occurrence; degree ordering suffices for taking the strong neighborhood
    attrs = _attrs().set_index("task")
    pos = {int(t): i for i, t in enumerate(tasks)}

    recs = []
    for t in tasks:
        ti = int(t)
        if ti in pos and ti in comm_of:
            i = pos[ti]
            nbrs = W.indices[W.indptr[i]:W.indptr[i + 1]]
            # keep only same-community neighbors, take the top 8 by job degree in descending order
            same = [int(tasks[j]) for j in nbrs if comm_of.get(int(tasks[j])) == comm_of[ti]]
            if len(same) < 2:
                continue
            same.sort(key=lambda x: -deg_job[pos[x]] if x in pos else 0)
            grp = [ti] + same[:8]
            if not (3 <= len(grp) <= 15):
                continue
            sub = attrs.reindex(grp)
            recs.append({
                "year": year, "seed": ti, "comm": comm_of[ti], "size": len(grp),
                "tasks": ";".join(str(x) for x in grp),
                "e_ord": ",".join(f"{v:.0f}" for v in sub["E_ord"].fillna(1)),
                "cod": ",".join(f"{v:.2f}" if pd.notna(v) else "na" for v in sub["codable"]),
                "hier": "|".join(str(h) for h in sub["hierarchy"].fillna("na")),
                "mod": "|".join(str(m) for m in sub["module"]),
                "n_mod": int(sub["module"].nunique()),
                "e2_share": float((sub["E_ord"] >= 2).mean()),
                "hier_n": int(sub["hierarchy"].nunique()),
            })
    df = pd.DataFrame(recs).drop_duplicates(subset=["tasks"])
    print(f"    task cluster candidates: {len(df)}", end="")
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed).reset_index(drop=True)
    print(f" → sampled {len(df)} (avg {df['size'].mean():.1f} tasks)")
    return df


def g3_firms(year: int, n_sample: int = 2000, seed=0) -> pd.DataFrame:
    """Criterion 3 object pool: firm × year. Antecedent fields = template-imprint features of the firm's job structure."""
    from jtagent import firm_panel

    fj = firm_panel.build_firm_year_jobs()
    adopt = pd.read_parquet(OUT / "results" / "rule_c_adoption.parquet")
    adopt["job"] = adopt["job"].astype(str)
    smap = adopt.set_index("job")

    sub = fj[fj["year"] == year]
    jobs = P.load_jobs(year)
    jobs["job"] = jobs["job"].astype(str)
    tset = dict(zip(jobs["job"], jobs["tasks"]))
    attrs = _attrs().set_index("task")

    recs = []
    for firm, g in sub.groupby("firm"):
        js = [j for j in g["job"].astype(str) if j in tset]
        if len(js) < 2:
            continue
        allt = set().union(*[set(tset[j]) for j in js])
        if not allt:
            continue
        a_sub = attrs.reindex(list(allt))
        struct = [smap["struct_sim"].get(j, np.nan) for j in js]
        xfirm = [smap["xfirm_struct_sim"].get(j, np.nan) for j in js]
        surf = [smap["surf_sim"].get(j, np.nan) for j in js]
        recs.append({
            "firm": firm, "year": year, "n_jobs": len(js),
            "mean_size": float(np.mean([len(tset[j]) for j in js])),
            "n_mod": int(a_sub["module"].nunique()),
            "e2_share": float((a_sub["E_ord"] >= 2).mean()),
            "struct_sim": float(np.nanmean(struct)) if np.isfinite(struct).any() else 0.0,
            "xfirm_sim": float(np.nanmean(xfirm)) if np.isfinite(xfirm).any() else 0.0,
            "surf_sim": float(np.nanmean(surf)) if np.isfinite(surf).any() else 0.0,
        })
    df = pd.DataFrame(recs)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed).reset_index(drop=True)
    print(f"    firm objects: {len(df)} (mean cross-firm template similarity {df['xfirm_sim'].mean():.3f})")
    return df


def build(year: int, n_sample: int = 2000):
    OBJ.mkdir(parents=True, exist_ok=True)
    a = _attrs()
    j = g1_jobs(a, year, n_sample)
    j.to_parquet(OBJ / f"g1_jobs_{year}.parquet", index=False)
    c = g2_communities(year, n_sample)
    c.to_parquet(OBJ / f"g2_comm_{year}.parquet", index=False)
    f3 = g3_firms(year, n_sample)
    f3.to_parquet(OBJ / f"g3_firms_{year}.parquet", index=False)
    print(f"[{year}] Criterion 1 objects {len(j)} (with E2 and with E0: {int((j.has_E2.eq(1)&j.has_E0.eq(1)).sum())}); "
          f"Criterion 2 task clusters {len(c)} (avg {c['size'].mean():.1f} tasks); Criterion 3 firms {len(f3)}")
    return j, c, f3


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for y in [int(x) for x in sys.argv[1:]] or [2020]:
        build(y)
