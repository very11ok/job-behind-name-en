"""Candidate task-pair generation, network structural features, formation/death labels.

For year T:
  - projection A(T): task x task 0/1 (co-occurrence within the same job)
  - candidate set = 2-hop and not yet bundled: cn>0, A=0
  - labels y=1[the pair is bundled in T+1]; y_new=1[the T+1 job bundling it is a new T+1 job]
  - graph features: deg, cn, jaccard, adamic-adar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SNAP = Path(r"D:\harness office\research\jobtask-agent\out\snapshots")


def load_edges(year: int) -> pd.DataFrame:
    return pd.read_parquet(SNAP / f"edges_{year}.parquet")


def load_jobs(year: int) -> pd.DataFrame:
    return pd.read_parquet(SNAP / f"jobs_{year}.parquet")


def bound_pair_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    out = set()
    grp = edges.groupby("job")["task"]
    for _, part in grp:
        arr = np.sort(part.unique())
        for i in range(len(arr)):
            a = int(arr[i])
            for b in arr[i + 1:]:
                out.add((a, int(b)) if a < int(b) else (int(b), a))
    return out


def build_projection(edges: pd.DataFrame):
    """Returns (tasks: sorted id array, tidx, A: adjacency csr int, deg: per-task job degree)."""
    tasks = np.sort(edges["task"].unique())
    tidx = {int(t): i for i, t in enumerate(tasks)}
    jobs, jidx_map = pd.factorize(edges["job"])
    r, c = jobs, edges["task"].map(tidx).to_numpy()
    B = sp.csr_matrix((np.ones(len(edges), np.float32), (r, c)),
                      shape=(len(jidx_map), len(tasks)))
    W = B.T @ B
    W.setdiag(0)
    W.eliminate_zeros()
    A = (W > 0).astype(np.int32).tocsr()
    deg_job = np.asarray((B > 0).astype(np.int32).sum(axis=0)).ravel()  # job degree
    deg_pair = np.asarray(A.sum(axis=1)).ravel()                        # co-occurrence neighborhood degree
    return tasks, tidx, A, deg_job, deg_pair


def common_neighbors(A: sp.spmatrix) -> sp.spmatrix:
    """Upper-triangular candidate matrix of unbundled pairs with cn>0; values = common-neighbor counts. int32 guards against overflow."""
    C = A @ A.T
    C = C.multiply(A == 0)
    C.setdiag(0)
    C.eliminate_zeros()
    return sp.triu(C, k=1).tocsr()


def adamic_adar(A: sp.spmatrix, deg_pair: np.ndarray) -> sp.spmatrix:
    w = 1.0 / np.log(np.maximum(deg_pair, 2.0))
    M = A @ sp.diags(w.astype(np.float32))
    AA = M @ A.T
    AA = AA.multiply(A == 0)
    AA.setdiag(0)
    AA.eliminate_zeros()
    return sp.triu(AA, k=1).tocsr()


def first_appear() -> pd.DataFrame:
    f = pd.read_csv(r"D:\wanted\exposure\new material\total\newjob\first_appear.csv")
    f = f.rename(columns={"num": "job", "date": "year"}).dropna(subset=["job"])
    f["year"] = f["year"].astype(int)
    return f
