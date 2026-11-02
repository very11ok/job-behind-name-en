"""Main backtest once annotation is ready: T=2019→2020, LLM semantic criteria vs lexicon vs pure-graph baseline.

Produces:
  out/results/agent_metrics_2019.csv    AP/AUC/P@k/lift for the three criteria (lex/llm) + baseline
  out/results/clash_2019.csv            logistic ablation clash (trained on 2018, tested on 2019)
  out/results/agent_preds_2019_sample.parquet  high-scoring candidate samples (for the LLM to supply why)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from jtagent import evaluate, features, model, rules
from jtagent.attrs import load_attrs

OUT = Path(r"D:\harness office\research\jobtask-agent\out")
sys.stdout.reconfigure(encoding="utf-8")


def _score(df: pd.DataFrame) -> pd.DataFrame:
    df = rules.add_scores(df)  # lexicon version
    return df


def _stamp(df: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Re-stamp the attribute columns onto the candidate-pair table (avoids the cache mixing up different attribute sources)."""
    a = attrs.set_index("task")
    df = df.copy()
    for col in ("E_ord", "ctx", "codable", "module", "hierarchy", "ctx_llm"):
        if col not in a.columns:
            continue
        df[f"{col}_i"] = a[col].reindex(df["ti"]).to_numpy()
        df[f"{col}_j"] = a[col].reindex(df["tj"]).to_numpy()
    df["E_gap"] = (df["E_ord_i"] - df["E_ord_j"]).abs()
    return df


def main(year: int = 2019, mode: str = "merge", covered_only: bool = False):
    if mode == "llm":
        covered_only = True
    # 1) read the cached candidate pairs + re-stamp with the specified attribute source
    df_cache = features.build_pairs_table(year)
    attrs_llm = load_attrs(mode=mode)
    df_llm = _stamp(df_cache, attrs_llm)
    mask = df_llm["ctx_i"].notna() & df_llm["ctx_j"].notna() if covered_only \
        else pd.Series(True, index=df_llm.index)
    if covered_only:
        before = len(df_llm)
        df_llm = df_llm[mask].copy()
        print(f"[covered-only] covered subgraph: {len(df_llm):,}/{before:,} pairs = "
              f"{len(df_llm)/max(before,1):.1%}; y=1 rate within coverage {df_llm['y'].mean():.4f}")
    print(f"[{year}] candidates={len(df_llm):,} positives={int(df_llm['y'].sum()):,} "
          f"y_new={int(df_llm['y_new'].sum()):,}")
    print("attr_src distribution:", attrs_llm["attr_src"].value_counts().to_dict())

    df_llm = _score(df_llm)
    names = list(rules.RULE_SCORES) + list(rules.BASELINE_SCORES)
    res_llm = evaluate.eval_frame(df_llm, names, target="y")
    res_llm["attr"] = {"llm": "LLM三守则", "merge": "llm+lex合并", "lex": "lex"}[mode] + \
                      ("(覆盖子图)" if covered_only else "")

    # 2) pure lexicon ablation (same covered subset)
    df_lex = _stamp(df_cache, load_attrs(mode="lex"))
    if covered_only:
        df_lex = df_lex[mask].copy()
    df_lex = _score(df_lex)
    res_lex = evaluate.eval_frame(df_lex, names, target="y")
    res_lex["attr"] = "lex消融"

    res = pd.concat([res_llm, res_lex])
    res.to_csv(OUT / "results" / f"agent_metrics_{year}.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))

    # 3) logistic clash: trained on T-1, tested on T. Three rows = pure graph / +lexicon semantics / +LLM semantics (same covered subset)
    try:
        def _frames(mode: str):
            tr = _stamp(features.build_pairs_table(year - 1), load_attrs(mode=mode))
            te = _stamp(features.build_pairs_table(year), load_attrs(mode=mode))
            if covered_only:
                m_tr = _stamp(features.build_pairs_table(year - 1), load_attrs(mode="llm"))
                m_te = _stamp(features.build_pairs_table(year), load_attrs(mode="llm"))
                tr = tr[(m_tr["ctx_llm_i"].notna() & m_tr["ctx_llm_j"].notna()).to_numpy()].copy()
                te = te[(m_te["ctx_llm_i"].notna() & m_te["ctx_llm_j"].notna()).to_numpy()].copy()
            return tr, te

        rows = []
        tr_l, te_l = _frames("lex")
        rows.append(model.clash_one(tr_l, te_l, False, "纯图+持续"))
        if covered_only:
            tr_i, te_i = _frames("llm")
            rows.append(model.clash_one(tr_i, te_i, True, "+LLM三守则语义"))
            rows.append(model.clash_one(tr_l, te_l, True, "+词表语义(对照)"))
        else:
            rows.append(model.clash_one(tr_l, te_l, True, "+三守则语义(词表)"))
        clash = pd.DataFrame(rows)
        clash.to_csv(OUT / "results" / f"clash_{year}.csv", index=False, encoding="utf-8-sig")
        print(clash.to_string(index=False))
    except Exception as e:
        print("clash failed:", repr(e))

    # 4) high-score samples (for LLM mechanism discrimination and why output)
    samp = pd.concat([
        df_llm.nlargest(300, "A_互补再捆绑").assign(tag="topA"),
        df_llm.nlargest(300, "B_层级解绑").assign(tag="topB"),
        df_llm.nlargest(300, "共邻").assign(tag="topCN"),
        df_llm.sample(600, random_state=1).assign(tag="random"),
    ]).drop_duplicates(subset=["ti", "tj", "tag"])
    texts = attrs_llm.set_index("task")["task_text"]
    samp["text_i"] = texts.reindex(samp["ti"]).to_numpy()
    samp["text_j"] = texts.reindex(samp["tj"]).to_numpy()
    samp.to_parquet(OUT / "results" / f"agent_preds_{year}_sample.parquet", index=False)
    print("sample:", samp.shape, "y=1 share:", round(samp["y"].mean(), 3))


if __name__ == "__main__":
    _y = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    _mode = sys.argv[2] if len(sys.argv) > 2 else "merge"
    _cov = len(sys.argv) > 3 and sys.argv[3] == "covered"
    main(_y, _mode, _cov)
