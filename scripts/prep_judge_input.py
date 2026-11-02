"""Generate batch inputs for the LLM antecedent judgment of the three criteria.

One text file per batch: out/judge_input/{rule}_batch_{XX}.txt:
  header = judgment protocol (antecedent definitions + output format), body = per-object rendered blocks.
Alignment keys: g1=job, g2=seed, g3=firm (consistent with labels_g*_2020.parquet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent")
OBJ = BASE / "out" / "objects"
JI = BASE / "out" / "judge_input"
YEAR = 2020
BATCH = 60

PROTO = {
    "g1": """[Judgment Protocol · Criterion (a) Complementary Re-bundling — Antecedent Judgment]
Unit: a job's task set (T=2020 snapshot).
Antecedent definition: the set contains both —
  ① "AI-absorbable high-exposure codable tasks": exposure E2, high codability, pure information processing, rules clear;
  ② "hard-to-absorb context-embedded tasks": depend on judgment/coordination/accountability/embodied context,
     low codability, exposure E0/E1.
Both task types coexist (① and ② both present) for the antecedent to hold=1; only one type or both weak → holds=0.
Meaning of p: given 2020 information, your estimate of the probability that this job will see "a new task combination with a high-low exposure pairing" in 2021-2023.
""",  # noqa: RUF001
    "g2": """[Judgment Protocol · Criterion (b) Hierarchical Unbundling — Antecedent Judgment]
Unit: a task cluster (a strong neighborhood centered on a core task in the task projection layer, T=2020).
Antecedent definition: tasks in the cluster satisfy —
  ① Same level: hierarchy (L0 routine execution / L1 standard problem-solving / L2 cross-functional coordination / L3 strategic judgment) highly concentrated;
  ② Predominantly codable: overall codability high, most tasks have clear rules;
  ③ Functional coherence: modules concentrated, task nature homogeneous — the whole cluster can be "carved" into narrower new jobs by "knowledge sinking".
All three points basically satisfied → holds=1; hierarchy mixed or many context-embedded tasks → holds=0.
Meaning of p: given 2020 information, your estimate of the probability that this cluster will "split into ≥2 narrower jobs, each carrying part of the cluster" in 2021-2023.
""",  # noqa: RUF001
    "g3": """[Judgment Protocol · Criterion (c) Mimetic Template Diffusion — Antecedent Judgment]
Unit: a firm × 2020 job structure.
Antecedent definition: the firm's job structure has "template residue" — surface similarity surf_sim is noticeably higher than structural similarity
(struct_sim = similarity to own past job structure, xfirm_sim = cross-firm task structure similarity):
  i.e. the firm's jobs resemble a popular template, but what they resemble is [wording] rather than [task structure] — structurally not truly isomorphic,
  indicating that the job description wording was copied from an external template and the task structure has not yet caught up.
When judging, compare the three values: the larger the residual of surf_sim relative to struct_sim/xfirm_sim, the more obvious the template residue, the more it favors holds=1;
the three values close (wording similarity explainable by structural similarity) → holds=0. Supplement with firm size and functional diversity background.
Meaning of p: given 2020 information, your estimate of the probability that the firm's "new postings with template adoption share above the network median" in 2021-2023.
""",  # noqa: RUF001
}

OUT_FMT = """
[Output] Strict JSON array, each element corresponding to one object block in the input, in the same order, output no other text:
[{"id":"<object id>","holds":<0|1>,"p":<0..1>,"why":"<≤30-char rationale>"}]
- id: copy the id from the first-line brackets of the object block.
- p: just give the bin: 0.05 (almost impossible) / 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 0.9.
"""


def _td() -> pd.DataFrame:
    td = pd.read_parquet(BASE / "out" / "snapshots" / "task_dict.parquet")
    td["task"] = td["task"].astype(int)
    texts = td.set_index("task")["task_text"].astype(str)
    return texts


def _trim(s: str, n: int = 90) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render_g1(texts) -> pd.DataFrame:
    df = pd.read_parquet(OBJ / f"g1_jobs_{YEAR}.parquet")
    blocks = []
    for _, r in df.iterrows():
        ts = [int(x) for x in str(r["tasks"]).split(";")]
        e = str(r["e_ord"]).split(",")
        cod = str(r["cod"]).split(",")
        ctx = str(r["ctx"]).split(",")
        mod = str(r["mod"]).split("|")
        lines = []
        for i, t in enumerate(ts):
            e_lab = {"0": "E0 low", "1": "E1 mid", "2": "E2 high"}.get(e[i].strip(), e[i])
            lines.append(f"  t{t} | {_trim(texts.get(t, '?'))} | {e_lab} | "
                         f"cod={cod[i]} ctx={ctx[i]} | {mod[i] if i < len(mod) else '?'}")
        blocks.append(f"[id={r['job']} | job task set {len(ts)} tasks | num functional modules {r['n_mod']}]\n"
                      + "\n".join(lines))
    return pd.DataFrame({"id": df["job"].astype(str), "block": blocks})


def render_g2(texts) -> pd.DataFrame:
    df = pd.read_parquet(OBJ / f"g2_comm_{YEAR}.parquet")
    blocks = []
    for _, r in df.iterrows():
        ts = [int(x) for x in str(r["tasks"]).split(";")]
        e = str(r["e_ord"]).split(",")
        cod = str(r["cod"]).split(",")
        hier = str(r["hier"]).split("|")
        mod = str(r["mod"]).split("|")
        lines = []
        for i, t in enumerate(ts):
            e_lab = {"0": "E0 low", "1": "E1 mid", "2": "E2 high"}.get(e[i].strip(), e[i])
            lines.append(f"  t{t} | {_trim(texts.get(t, '?'))} | {e_lab} | "
                         f"cod={cod[i]} | {hier[i] if i < len(hier) else '?'} | "
                         f"{mod[i] if i < len(mod) else '?'}")
        blocks.append(f"[id={r['seed']} | task cluster {len(ts)} tasks | E2 share {r['e2_share']:.2f} | "
                      f"num hierarchy levels {r['hier_n']} | num modules {r['n_mod']}]\n" + "\n".join(lines))
    return pd.DataFrame({"id": df["seed"].astype(str), "block": blocks})


def render_g3(texts) -> pd.DataFrame:
    df = pd.read_parquet(OBJ / f"g3_firms_{YEAR}.parquet")
    blocks = []
    for _, r in df.iterrows():
        resid = float(r["surf_sim"]) - max(float(r["struct_sim"]), float(r["xfirm_sim"]))
        blocks.append(
            f"[id={r['firm']} | firm job structure 2020]\n"
            f"  num jobs {r['n_jobs']} | mean task set {r['mean_size']:.1f} tasks | num functional modules {r['n_mod']}"
            f" | high-exposure E2 task share {r['e2_share']:.2f}\n"
            f"  structural similarity (own past) struct_sim={r['struct_sim']:.3f}"
            f" | cross-firm structural similarity xfirm_sim={r['xfirm_sim']:.3f}"
            f" | surface text similarity surf_sim={r['surf_sim']:.3f}"
            f" | residual surf-struct={resid:+.3f}")
    return pd.DataFrame({"id": df["firm"].astype(str), "block": blocks})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    JI.mkdir(parents=True, exist_ok=True)
    texts = _td()
    total_batches = 0
    for rule, fn in (("g1", render_g1), ("g2", render_g2), ("g3", render_g3)):
        df = fn(texts)
        header = PROTO[rule] + OUT_FMT
        n = 0
        for i in range(0, len(df), BATCH):
            part = df.iloc[i : i + BATCH]
            body = header + "\n".join(part["block"])
            (JI / f"{rule}_batch_{n:03d}.txt").write_text(body, encoding="utf-8")
            n += 1
        total_batches += n
        print(f"{rule}: {len(df)} objects → {n} batches (≤{BATCH} per batch)")
    print(f"{total_batches} batches in total → {JI}")


if __name__ == "__main__":
    main()
