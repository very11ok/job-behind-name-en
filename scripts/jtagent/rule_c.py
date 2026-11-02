"""Rule (c) mimetic template diffusion: decomposition of structural similarity and surface-text similarity for new job bundles.

Core idea (DiMaggio & Powell 1983): organizations copy existing structures under uncertainty.
Mimetic adoption = the part of surface-text similarity that still remains after controlling for
task-structural similarity.

For each bundle newly appearing in year T (first_appear==T):
  struct_sim = max Jaccard against all old bundles (task sets)
  surf_sim   = job-name surface similarity against all old bundles (char bigram TFIDF cosine)
  Outcome indicators:
    - struct_sim ≥ 0.8 → template adoption that is a near-verbatim copy
    - 0.4 ≤ struct_sim < 0.8 → improved copying
    - surf_resid = surf_sim - bin_meansurf(struct_sim) → pure surface-imitation residual
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from . import pairs as P

OUT = Path(r"D:\harness office\research\jobtask-agent\out")


def _all_bundles() -> pd.DataFrame:
    """All-period bundle table: job, tasks(set), name, firm, first_year."""
    fa = P.first_appear().rename(columns={"year": "first_year"})
    rows = []
    for y in range(2016, 2027):
        j = P.load_jobs(y)
        rows.append(j[["job", "tasks", "name", "firm", "n_postings"]])
    b = pd.concat(rows).drop_duplicates("job")
    b = b[b["name"].astype(str).str.len() > 0]
    b["tasks"] = b["tasks"].map(lambda x: frozenset(int(t) for t in x))
    b = b.merge(fa, on="job", how="left")
    b["first_year"] = b["first_year"].fillna(2016).astype(int)
    return b


def _max_jaccard(new: pd.DataFrame, old: pd.DataFrame):
    """For each new bundle returns (max_jaccard, argmax_old row index, cross_firm_max_jaccard).

    cross_firm: max Jaccard after excluding the same firm (within-firm reuse does not count as diffusion, only as imitation).
    """
    inv = defaultdict(list)
    for k, ts in enumerate(old["tasks"]):
        for t in ts:
            inv[t].append(k)
    old_tasks = old["tasks"].tolist()
    old_firms = old["firm"].tolist()
    best = np.zeros(len(new))
    argmax_k = np.full(len(new), -1, dtype=np.int64)
    best_x = np.zeros(len(new))
    for i, (ts, f) in enumerate(zip(new["tasks"], new["firm"])):
        if not ts:
            continue
        cand = defaultdict(int)
        for t in ts:
            for k in inv.get(t, ()):
                cand[k] += 1
        b, bk, bx = 0.0, -1, 0.0
        ns = len(ts)
        for k, inter in cand.items():
            no = len(old_tasks[k])
            jac = inter / (ns + no - inter)
            if jac > b:
                b, bk = jac, k
            # must exclude the same firm: that old bundle's first-appearing firm != the new posting's firm
            if old_firms[k] != f and jac > bx:
                bx = jac
        best[i], argmax_k[i], best_x[i] = b, bk, bx
    return best, argmax_k, best_x


def _max_surface(new: pd.DataFrame, old: pd.DataFrame, vect=None) -> np.ndarray:
    """Max char-bigram TFIDF cosine of job names (against the old pool)."""
    if vect is None:
        vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
        vect.fit(old["name"].astype(str))
    V_old = vect.transform(old["name"].astype(str))
    V_new = vect.transform(new["name"].astype(str))
    S = V_new @ V_old.T
    return np.asarray(S.max(axis=1).todense()).ravel(), vect


def run_rule_c(years: range = range(2017, 2026)) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundles = _all_bundles()
    vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
    vect.fit(bundles["name"].astype(str))  # vocabulary taken over the whole period to avoid yearly drift; the transform uses the <T pool
    recs = []
    for y in years:
        new = bundles[bundles["first_year"] == y].reset_index(drop=True)
        old = bundles[bundles["first_year"] < y].reset_index(drop=True)
        if not len(new) or not len(old):
            continue
        st, st_k, st_x = _max_jaccard(new, old)
        sf = _max_surface(new, old, vect=vect)[0]
        tmpl_jobs = old["job"].to_numpy()
        r = pd.DataFrame({
            "job": new["job"], "year": y, "firm": new["firm"], "name": new["name"],
            "struct_sim": st,
            "xfirm_struct_sim": st_x,
            "surf_sim": sf,
            "tmpl_job": [tmpl_jobs[k] if k >= 0 else None for k in st_k],
        })
        recs.append(r)
        print(f"[rule_c] {y}: new={len(r)} share≥0.8={(st>=.8).mean():.3f} "
              f"x-firm≥0.8={(st_x>=.8).mean():.3f}")
    ad = pd.concat(recs)
    ad["tmpl_firm"] = ad["tmpl_job"].map(bundles.set_index("job")["firm"])
    ad["tmpl_name"] = ad["tmpl_job"].map(bundles.set_index("job")["name"])
    ad["cross_firm"] = (ad["tmpl_firm"] != ad["firm"]).astype(int)
    ad["struct_bin"] = pd.cut(ad["struct_sim"], bins=[0, .2, .4, .6, .8, 1.000001],
                              labels=False, include_lowest=True)
    ad["surf_resid"] = ad["surf_sim"] - ad.groupby(["year", "struct_bin"])["surf_sim"].transform("mean")
    res_dir = OUT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    ad.to_parquet(res_dir / "rule_c_adoption.parquet", index=False)

    def share(x, lo, hi):
        return float(((x >= lo) & (x < hi)).mean())
    tr = ad.groupby("year").apply(
        lambda g: pd.Series({
            "n_new": len(g),
            "复制≥0.8": share(g["struct_sim"], .8, 2),
            "改良0.4-0.8": share(g["struct_sim"], .4, .8),
            "新结构<0.4": share(g["struct_sim"], 0, .4),
            "跨企复制≥0.8": share(g.loc[g["cross_firm"] == 1, "struct_sim"], .8, 2),
            "跨企改良0.4-0.8": share(g.loc[g["cross_firm"] == 1, "struct_sim"], .4, .8),
            "surf_resid>0": float((g["surf_resid"] > 0).mean()),
            "surf_resid 均值": round(float(g["surf_resid"].mean()), 4),
            "surf_resid P90": round(float(g["surf_resid"].quantile(0.9)), 4),
        }), include_groups=False).reset_index()
    tr.to_csv(res_dir / "rule_c_trend.csv", index=False, encoding="utf-8-sig")
    return ad, tr
