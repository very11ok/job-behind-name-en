"""Computable scoring of the three re-bundling criteria (df must contain the columns produced by features.py).

The attribute source is swappable: lexicon (ablation baseline) or LLM annotation (main criterion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ctx_low(df: pd.DataFrame) -> pd.Series:
    """Context-embeddedness of the low-exposure end."""
    take_i = df["E_ord_i"] <= df["E_ord_j"]
    return df["ctx_i"].where(take_i, df["ctx_j"])


def score_A(df: pd.DataFrame) -> pd.Series:
    """(a) complementarity re-bundling: large E gap x context-embeddedness of the low-end task.

    Theoretical signature: after codable (E2) tasks are absorbed, they are newly co-bundled with
    hard-to-absorb judgment/coordination/accountability/embodied (E0, high-ctx) tasks.
    """
    mix = df["E_gap"] / 2.0
    return mix * _ctx_low(df)


def score_B(df: pd.DataFrame) -> pd.Series:
    """(b) hierarchical unbundling: same module x same exposure x codable x neighborhood cohesion -> narrow homogeneous new combinations.

    Direction opposite to (a): a low E gap adds points instead.
    """
    same_mod = (df["module_i"] == df["module_j"]).astype(float)
    hom = 1.0 - df["E_gap"] / 2.0
    cod = np.sqrt(df["codable_i"].clip(0) * df["codable_j"].clip(0))
    coh = 0.5 + 0.5 * df["jaccard"]
    return same_mod * hom * cod * coh


def score_PERSIST(df: pd.DataFrame) -> pd.Series:
    return df["bind_TM1"]


def score_PA(df: pd.DataFrame) -> pd.Series:
    return np.log1p(df["pa"])  # log-transform so extreme values do not dominate the ranking


def score_CN(df: pd.DataFrame) -> pd.Series:
    return df["cn"]


def score_JAC(df: pd.DataFrame) -> pd.Series:
    return df["jaccard"]


# NOTE: Chinese data schema kept verbatim (dict keys become DataFrame column names / output values)
RULE_SCORES = {"A_互补再捆绑": score_A, "B_层级解绑": score_B}
BASELINE_SCORES = {"持续性": score_PERSIST, "优先连接": score_PA,
                   "共邻": score_CN, "jaccard": score_JAC}


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    for name, fn in {**RULE_SCORES, **BASELINE_SCORES}.items():
        df[name] = fn(df)
    return df
