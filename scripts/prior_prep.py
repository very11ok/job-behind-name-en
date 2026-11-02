"""Let the LLM judge the antecedent + probability bin for each object. Three prompts correspond to the three criteria.

Output JSON, one line per object:
  g1: {"job": "...", "prior": <0|1>, "p": <0..1>, "level": <"low"|"mid"|"high">, "why": "..."}
  g2: {"seed": <int>, "prior": ..., "p": ..., "level": ..., "why": ...}
  g3: {"firm": "...", "prior": ..., "p": ..., "level": ..., "why": ...}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
OUT = Path(r"D:\harness office\research\jobtask-agent\out")

SYS_G1 = """You are an organizational-economics judge. Given a job's task set (each task carries: O*NET task text,
GPT-4 exposure level E0/E1/E2, codability cod 0-2, context embedding ctx 0-1, functional module).

[Criterion (a) Complementary Re-bundling] After AI absorbs codable tasks, the remaining hard-to-absorb tasks (judgment/coordination/accountability/embodied)
are re-bundled with the absorbed high-exposure tasks.
Antecedent = the job's task set **simultaneously contains**: AI-absorbable high-exposure codable tasks + hard-to-absorb context-embedded tasks.

Judge whether this job has this antecedent, and predict whether it will evolve into a "hybrid new combination" (the task set shows
a new pairing of high-exposure × low-exposure). Output one line of JSON:
{"job": "<job id>", "prior": <1=antecedent present 0=absent>, "p": <0..1 probability of a hybrid new combination>,
 "level": "<low|mid|high>", "why": "<≤25 chars, identifying the antecedent basis>"}
Base only on the given information; strict JSON array output."""

SYS_G2 = """You are an organizational-economics judge. Given a "task cluster" (a group of tightly connected tasks in the projection network,
each carrying exposure level E, codability cod, knowledge hierarchy L0-L3, functional module).

[Criterion (b) Hierarchical Unbundling] After knowledge acquisition cost falls, the problem-solving span narrows, and codable tasks at the same knowledge level
are split into narrower, more homogeneous jobs.
Antecedent = tasks in the cluster **at the same level, codable, functionally homogeneous** (i.e. a knowledge module that can be neatly carved out).

Judge whether this cluster has the antecedent, and predict whether it will **split** into several narrower new task sets. Output one line of JSON:
{"seed": <int seed task id>, "prior": <1|0>, "p": <0..1 probability of splitting>,
 "level": "<low|mid|high>", "why": "<≤25 chars>"}
strict JSON array output."""

SYS_G3 = """You are an organizational-economics judge. Given a summary of a firm's job structure for the year
(number of jobs, average number of tasks, task functional-module distribution, cross-firm template adoption rate, exposure structure).

[Criterion (c) Mimetic Template Diffusion] Under uncertainty, organizations copy seemingly successful structures.
Antecedent = the firm's structure **already has cross-firm template imprinting** (its job combinations are highly isomorphic with external templates).

Judge whether this firm has the antecedent, and predict whether the template adoption share in its future new postings will exceed the industry contemporaneous level.
Output one line of JSON:
{"firm": "<firm code>", "prior": <1|0>, "p": <0..1>, "level": "<low|mid|high>", "why": "<≤25 chars>"}
strict JSON array output."""


def prep_batches(rule: str, year: int = 2020, batch: int = 40) -> list[Path]:
    """Slice the object pool into LLM input batches (CSV)."""
    d_in = OUT / "prior_in" / rule / str(year)
    d_in.mkdir(parents=True, exist_ok=True)
    if rule == "g1":
        df = pd.read_parquet(OUT / "objects" / f"g1_jobs_{year}.parquet")
        cols = ["job", "size", "e_ord", "cod", "ctx", "mod"]
    elif rule == "g2":
        df = pd.read_parquet(OUT / "objects" / f"g2_comm_{year}.parquet")
        cols = ["seed", "size", "e_ord", "cod", "hier", "mod"]
    else:
        p = OUT / "objects" / f"g3_firms_{year}.parquet"
        df = pd.read_parquet(p)
        cols = list(df.columns)
    df = df[cols].reset_index(drop=True)
    paths = []
    for k in range(0, len(df), batch):
        f = d_in / f"batch_{k//batch:03d}.csv"
        df.iloc[k:k + batch].to_csv(f, index=False, encoding="utf-8-sig")
        paths.append(f)
    (d_in / "meta.json").write_text(json.dumps(
        {"rule": rule, "year": year, "n": len(df), "batches": len(paths)}, ensure_ascii=False), "utf-8")
    print(f"[{rule}] {len(df)} objects → {len(paths)} batches")
    return paths


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for r in sys.argv[1:] or ["g1", "g2"]:
        prep_batches(r, 2020)
