"""Ablation clash: pure graph structure vs +three-criteria semantics; logistic trained on T-1, evaluated on T.

Outputs AP/AUC/lift/Brier for the two feature sets each year. This is the key test of
"whether the LLM semantic criteria add any increment".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

BASE_FEATS = ["cn_log", "jaccard", "degp_log_i", "degp_log_j", "bind_TM1"]
RULE_FEATS = ["E_gap_norm", "ctx_low", "same_mod", "cod_geom"]

OUT = Path(r"D:\harness office\research\jobtask-agent\out\results")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["cn_log"] = np.log1p(x["cn"])
    x["degp_log_i"] = np.log1p(x["degp_i"])
    x["degp_log_j"] = np.log1p(x["degp_j"])
    x["E_gap_norm"] = x["E_gap"] / 2.0
    take_i = x["E_ord_i"] <= x["E_ord_j"]
    x["ctx_low"] = x["ctx_i"].where(take_i, x["ctx_j"])
    x["same_mod"] = (x["module_i"] == x["module_j"]).astype(float)
    x["cod_geom"] = np.sqrt(x["codable_i"].clip(0) * x["codable_j"].clip(0))
    return x


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    npos = max(int(y.sum()), 1)
    order = np.argsort(-p)
    ys = y[order]
    ap = float((ys * ys.cumsum() / np.arange(1, n + 1)).sum() / npos)
    ranks = np.empty(n)
    ranks[order] = np.arange(n, 0, -1)   # high scores get large ranks
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * (n - npos), 1))
    brier = float(np.mean((p - y) ** 2))
    m = max(n // 100, 1)
    lift1 = float(ys[:m].mean() / max(y.mean(), 1e-12))
    return {"AP": round(ap, 5), "AUC": round(auc, 4), "Brier": round(brier, 6),
            "lift@1%": round(lift1, 2)}


def clash(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str = "y",
          C: float = 0.5) -> pd.DataFrame:
    """train = T-1 (features from the T-1 network), test = T. Returns two rows of metrics."""
    tr, te = _prep(df_train), _prep(df_test)
    rows = []
    # NOTE: Chinese data values kept verbatim (model labels written to the output DataFrame/CSV)
    for label, feats in [("纯图+持续", BASE_FEATS), ("+三守则语义", BASE_FEATS + RULE_FEATS)]:
        xf = tr[feats].fillna(0).to_numpy()
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(xf, tr[target].to_numpy())
        p = m.predict_proba(te[feats].fillna(0).to_numpy())[:, 1]
        r = _metrics(te[target].to_numpy(), p)
        r["model"] = label
        rows.append(r)
    return pd.DataFrame(rows)


def clash_one(df_train: pd.DataFrame, df_test: pd.DataFrame, use_rules: bool,
              label: str, target: str = "y", C: float = 0.5) -> dict:
    """Single clash: pure-graph baseline, or with the three-criteria semantic features added."""
    tr, te = _prep(df_train), _prep(df_test)
    feats = BASE_FEATS + (RULE_FEATS if use_rules else [])
    m = LogisticRegression(C=C, max_iter=1000)
    m.fit(tr[feats].fillna(0).to_numpy(), tr[target].to_numpy())
    p = m.predict_proba(te[feats].fillna(0).to_numpy())[:, 1]
    r = _metrics(te[target].to_numpy(), p)
    r["model"] = label
    return r
