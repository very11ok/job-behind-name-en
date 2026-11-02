"""Merge LLM task annotation batches -> out/results/anno_<YEAR>.parquet, with validation and fallback-relabeling strategy output.

Usage: python merge_anno.py 2019
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent\out")
REQ = ["id", "codability", "judgment", "coordination", "accountability",
       "embodied", "hierarchy", "module"]
MODS = {"生产执行", "质量检验", "研发技术", "工艺工程", "设备维护", "销售", "市场",
        "采购供应链", "物流仓储", "人力资源", "财务会计", "行政法务", "IT数据",
        "EHS安全", "客户服务", "综合管理", "其他"}


def merge_year(year: int):
    d_in, d_out = BASE / "anno_input" / str(year), BASE / "anno_output" / str(year)
    meta = json.loads((d_in / "meta.json").read_text("utf-8"))
    rows, bad = [], []
    for k in range(meta["n_batches"]):
        f = d_out / f"batch_{k:03d}.json"
        if not f.exists():
            bad.append(f.stem)
            continue
        try:
            arr = json.loads(f.read_text(encoding="utf-8-sig"))
            for x in arr:
                if all(key in x for key in REQ):
                    rows.append(x)
        except Exception:
            bad.append(f.stem)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["module"] = df["module"].where(df["module"].isin(MODS), "其他")
    df.to_parquet(BASE / "results" / f"anno_{year}.parquet", index=False)

    # coverage
    covered = df["id"].nunique()
    print(f"[{year}] annotated {len(df)} records, covering {covered}/{meta['n_tasks']} tasks; bad batches {len(bad)}: {bad[:10]}")

    # agreement with GPT-4 exposure grades (calibration evidence)
    td = pd.read_parquet(BASE / "snapshots" / "task_dict.parquet")
    m = df.merge(td[["task", "E_ord", "E"]], left_on="id", right_on="task")
    if len(m):
        ct = pd.crosstab(m["E_ord"], m["codability"], normalize="index").round(3)
        print("codability × GPT-4 E_ord row-normalized crosstab:")
        print(ct.to_string())
    return bad


if __name__ == "__main__":
    for y in [int(x) for x in sys.argv[1:]] or [2019]:
        merge_year(y)
