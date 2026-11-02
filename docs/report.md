# The Job Before the Name: first backtest of the three-criteria agent on the new-materials job–task network

v1.0 · 2026-10-28 · Main setting **T=2019 → 2020**; data: 2016–2026 job-posting text of Chinese new-materials listed companies

> **Status note (2026-10-31)**: This report is a **historical document** from the T=2019 pairwise-backtest stage; its conclusions are as of the evidence available at that time.
> The statement "(b) is the only criterion that passed the tests in this round" has since been revised by the later T=2020 three-criteria antecedent-judgment experiment:
> after a confounding-structure diagnosis, a threshold scan (hier_n≥3), and a directional reconstruction, the inverse relationship of criterion b turned strongly positive
> (lift ratios 2.34/2.59/2.49), and after the vacuous shell of criterion c was replaced with objective antecedents (n_jobs/n_mod), all three windows became significant
> (2.98/3.26/3.06). For the latest conclusions see README and `docs/results/`. The "three-layer" framework of §2, the agent
> composition of §3, and the placebo-clash design of §5.3 are carried over into the later experiments.

---

## 0. Goal (in one sentence)

Build an agent that obeys the three re-bundling criteria: it reads the T-year job–task bipartite network snapshot, takes
**(a) complementarity re-bundling / (b) hierarchical unbundling / (c) mimetic template diffusion** as its generative criteria,
**with the LLM performing all semantic and similarity judgments** (the agent itself + sub-agents annotating in parallel),
while network structure quantities, candidate generation, and scoring backtests are done by deterministic algorithms; the predictions are checked against the actual T+1 network and four structural baselines,
and it is forced to pass a "placebo control" test: the semantic increment must come from LLM judgments, not from arbitrary semantic features.

## 1. Data and replication

| Year | Paper Table 1 jobs/tasks/edges | This replication | Agreement |
|---|---|---|---|
| 2020 | 8318 / 6110 / 42007 | 8311 / 6109 / 41990 | 99.96% |
| 2023 | 6396 / 5772 / 35551 | 6396 / 5772 / 35551 | **100%** |
| 2026 | 3446 / 4601 / 21269 | 3446 / 4601 / 21269 | **100%** |
| Other years | — | — | differences ≤3% (where task sets disagree within the same job J, take the first occurrence; 84–139 per year) |

- Edge table: `total/merged/YYYY.csv`; `jobmatch` = job J id; `taskmatch` = semicolon-separated O*NET task ids.
- Task semantics: `idmapping.csv` (GPT-4 exposure levels E0/E1/E2) + this project's LLM annotations (see §3).
- Cache: `out/snapshots/{jobs,edges}_YYYY.parquet`, `task_dict.parquet`.

## 2. Unit of analysis: three layers, each mechanism in its own layer

| Layer | Role | Mechanism | Judgment event |
|---|---|---|---|
| **Job bundle** (a set of tasks) | the object being re-bundled | (a) | new bundles enter carrying "high×low exposure mixed" task sets |
| **Firm × year** | the decision-making entity | (b) | within-firm narrowing of the problem-solving span / job-split events |
| **Network (field)** | the stage for diffusion and testing | (c) | cross-firm isomorphic template adoption + surface residuals |

Rationale: the complementarity of (a) is a property of task combinations **within the same production unit**; the knowledge hierarchy of (b) grows **inside the firm**,
and aggregate network totals cannot show who is splitting jobs; the imitation of (c) is meaningful only when it carries **firm identity** (reuse within the same firm does not count as diffusion).
The pairwise backtest (§5) is run at the network layer, because the predictor (an external analyst/agent) can only see the full-network snapshot of year T.

## 3. Composition of the agent

The **constitution** (`jtagent/prompts.py`, version v0.2) has three parts, each containing the knowledge-cutoff discipline and the "structured output only" constraint:

1. **Task-annotation constitution**: judge every O*NET task into
   `codability(0–2)`, `judgment/coordination/accountability/embodied(0/1)`,
   `hierarchy(L0–L3, Garicano's knowledge-based hierarchy)`, `module(17 functional-module classes)`.
2. **Task-pair judgment constitution**: for each pair, give the co-bundling probability p, a mechanism label `A/B/C/none`, and a rationale of ≤30 characters.
3. **Job-generation constitution**: for high-exposure jobs, output the restructured task combination for T+1 (constrained to take task ids only from the candidate pool).

**Execution**: the LLM of this session is the agent itself; annotation at scale is carried out in waves by sub-agents dispatched via `workflow`
(48 tasks per batch, concurrency ≤6), and the few batches whose sub-agents repeatedly failed were fallback-relabeled directly by the agent itself.
The 5,721 tasks active in 2019 → **all annotated (120/120 batches, QC passed, 100%)**,
covering **3,599,363** candidate pairs (the full set).

**Three tiers of attribute sources**: `lex` (English lexicons, for ablation/placebo), `llm` (LLM annotations), `merge` (LLM takes precedence, lexicon backfills).

## 4. Backtest protocol

- **Candidate pool**: pairs that are 2-hop apart in the T-year task-projection network and not yet bundled. Full set at T=2019: 3,599,363 pairs.
- **Labels**: `y=1` if the pair is bundled by any active job in year T+1; `y_new=1` if it is carried by jobs **newly entering** in T+1.
  Full set at T=2019: 46,485 positives, of which **97% are carried by newly entering bundles** → the prediction target = the new co-bundling brought by new jobs.
- **Baseline family**: persistence (bundled in T-1), preferential attachment, common neighbors, jaccard, and (§5.3) a graph-only logistic model.
- **Metrics**: AP, ROC-AUC, precision@k (k = number of positives), lift@1%, Brier.
- **Leakage prevention**: all features use only information from ≤T; annotation uses only task text (O*NET statements, which predate the sample period);
  the agent never outputs occupation names.

## 5. Results

### 5.1 Signature evidence for the three criteria (the part that does not require the LLM, 2016–2026)

**(a) Complementarity re-bundling: ✓ present, and accelerating after the generative-AI shock**

| Year | Number of new bundles | Average number of tasks | Within-bundle E variance | High–low exposure mixing share (E0×E2) |
|---|---|---|---|---|
| 2017 | 14,205 | 4.90 | 0.463 | 0.575 |
| 2020 | 6,924 | 5.35 | 0.458 | 0.573 |
| 2022 | 7,792 | 5.98 | 0.503 | 0.642 |
| 2025 | 11,946 | 6.26 | 0.500 | 0.658 |

New jobs are getting broader, more internally differentiated, and increasingly "mixed" between high-exposure and context-embedded tasks. (`signature_trend.csv`)

**(b) Hierarchical unbundling: ✗ no narrowing seen (this industry, this window)**

Firm–year panel (3,790 firm–years): average number of tasks per job 2.83→5.78, firm task universe per job 2.54→5.28,
within-job E variance 0.243→0.456, **all rising**; job-split events 35–230 per year, no trend.
→ Garicano's span narrowing is swamped by the complementary reinforcement of (a). (`firm_year_stats.csv`, `split_events.csv`)

**(c) Mimetic template diffusion: ✓ about 1/3 of new jobs are products of existing templates**

Structurally similar ≥0.8 (near-verbatim): 1.3–4.0% within firms, 0.5–1.2% across firms;
0.4–0.8 (improved copying) stable at ~33%; the share with surface-similarity residual >0 rose from 0.41(2017) to 0.58(2021+).
(`rule_c_trend.csv`, `rule_c_adoption.parquet`)

### 5.2 Pairwise backtest (T=2019→2020, full set 3,599,363 pairs, 46,485 positives)

| Score | AP | AUC | P@k | lift@1% |
|---|---|---|---|---|
| Common neighbors (strongest single baseline) | 0.1081 | 0.7912 | 0.1832 | 15.65 |
| Preferential attachment | 0.0887 | 0.8112 | 0.1571 | 13.34 |
| jaccard | 0.0238 | 0.6213 | 0.0376 | 3.01 |
| Persistence | 0.0454 | 0.5951 | 0.1439 | 11.03 |
| B4 module×same-exposure×cohesive · **LLM** | 0.0199 | **0.5374** | **0.0543** | 4.61 |
| B4 · lexicon | 0.0157 | 0.5176 | 0.0316 | 2.76 |
| B1 module×homogeneous×codable×cohesive · **LLM** | 0.0166 | **0.5309** | **0.0375** | 3.09 |
| B1 · lexicon | 0.0136 | 0.5075 | 0.0165 | 1.38 |
| A complementarity re-bundling (best variant) · LLM | 0.0125 | 0.4855 | 0.0132 | 1.07 |
| A · lexicon | 0.0128 | 0.4947 | 0.0141 | 1.04 |

**How to read this**: on the same candidate subset, once (b) uses LLM judgments the AUC goes 0.5075→0.5309 and 0.5176→0.5374,
and P@k **doubles, up to 2.3×** (0.0165→0.0375; 0.0316→0.0543) → the LLM's module/hierarchy judgments bring usable structural signal;
(a) shows no signal with either the LLM or the lexicon (AUC<0.5, i.e., the rule barely holds at the "task-pair" level).

### 5.3 Three-way clash (logistic: 2018 train → 2019 test, full set, same subset)

| Model | AP | AUC | Brier | lift@1% |
|---|---|---|---|---|
| Graph-only + persistence (null model) | 0.12482 | 0.8267 | 0.012239 | 17.10 |
| **+ lexicon semantics (placebo control)** | 0.12441 | 0.8271 | 0.012236 | 17.08 |
| **+ LLM three-criteria semantics** | **0.12926** | **0.8354** | **0.012211** | **17.52** |

**This is the most important result of this round**: the lexicon version is indistinguishable from the null model (the placebo passes; AP is even slightly lower),
while the LLM version improves all four metrics at once (AUC +0.0087, AP +0.0044, Brier −0.000028, lift +0.42).
The increment comes from the LLM judgments themselves, not from "just adding some semantic features".

### 5.4 Feature-by-feature ablation: what carries the increment (key diagnosis)

2018 train → 2019 test, adding semantic features one at a time:

| Feature added | AP | AUC | Brier | lift@1% |
|---|---|---|---|---|
| Graph-only baseline | 0.12482 | 0.8267 | 0.012239 | 17.10 |
| + E_gap_norm (exposure gap, the core of (a)) | 0.12490 | 0.8275 | 0.012238 | 17.10 |
| + ctx_low (context-embeddedness, the other half of (a)) | 0.12451 | **0.8257** | 0.012238 | 17.08 |
| + same_mod (same functional module, the core of (b)) | **0.12899** | **0.8350** | **0.012208** | **17.49** |
| + cod_geom (codability) | 0.12449 | 0.8264 | 0.012239 | 17.07 |
| + all semantics | 0.12926 | 0.8354 | 0.012211 | 17.52 |

Comparison (lexicon attributes, same subset): + same_mod only 0.8282→0.8293 (+0.0011); all semantics 0.8302.

**Conclusion**: the increment is **carried almost entirely by `same_mod`** (+0.0083), and the LLM's functional-module labels are significantly better than the O*NET SOC-derived modules
(+0.0083 vs +0.0011); `E_gap`, `ctx_low`, and `cod_geom` are near zero or even negative.
That is: the current LLM's contribution is **better functional-semantic classification** (the measurement layer), not yet the mechanisms of (a)/(b) themselves.

### 5.5 Agent blind-judgment pilot (n=48, ground truth unseen)

I blind-judged the 48 pairs under the three-criteria constitution: AUC **0.888**, top-5 hits 3/5; for comparison, common neighbors 0.847 (1/5),
preferential attachment 0.814, persistence 0.758, lexicon A 0.209 / lexicon B 0.400.
Observations: of the 5 new co-bundlings that materialized, **all** were judged by me as B (same-module homogeneous); none of the 7 A judgments I made materialized;
**C was never assigned** — mimetic imitation cannot be recognized from task-pair semantics and must rely on firm/template context (consistent with the surface-residual design of (c)).
(`pair_judge_llm.json`, `pair_judge_truth.parquet`)

## 6. Calibration findings (a by-product, but an important one)

The LLM's `codability` (whether it can be replaced by machines) and GPT-4's `E` (exposure) diverge systematically:

| | codability=0 | =1 | =2 |
|---|---|---|---|
| E0 | 45.2% | 51.9% | 2.9% |
| E1 | 6.1% | 60.6% | 33.3% |
| **E2** | **24.4%** | 68.0% | **7.6%** |

Nearly a quarter of E2 tasks are judged "highly dependent on on-site presence/relationships/accountability", and only 7.6% are explicitly judged purely codable.
**"The LLM can influence" ≠ "machines can replace"** — this supports the proposal's dichotomous design of treating exposure as shock pressure and context-embeddedness as the condition for retention/re-bundling.

## 7. Conclusions and boundaries

1. **(b) is the only criterion that passed the tests in this round**: the LLM's module/hierarchy judgments have a real increment in pairwise prediction (P@k doubles).
2. **(a) does not hold at the "task-pair" level, but holds at the "job composition" level**: the exposure mixing of new jobs does rise,
   but the large majority of newly added task pairs are still **same-module** (the B pattern). Explanation: high×low exposure mixing happens
   *within modules* (the same module contains both codable tasks and contextual tasks), so the E gap is a property of the module rather than of the "new pair";
   in pairwise prediction, using the E gap gives no leverage, using modules does.
   → **Fix**: (a) should be rewritten as **job-level mixing prediction** (predicting the E variance/mixing share of a new job given its modules),
   rather than task-pair-level link prediction.
3. **(c) must be identified at the firm/template layer**; task-pair semantics cannot identify imitation (proven in §5.4).
4. **Industry specificity**: the absence of (b) may be a feature of the new-materials industry (process/on-site intensive, strongly complementary);
   reruns on samples from other industries are needed to tell whether this is a theoretical problem or a contextual one.
5. **Coverage limitation**: LLM annotations cover 84.1% of tasks and 71% of candidate pairs; the missingness is random (shuffled within batches),
   comparisons within subgraphs are valid, but the full-set numbers remain to be filled in (19 batches remaining).

## 8. Reproducing the Results

```bash
PY="C:\Users\liuqin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
cd "D:\harness office\research\jobtask-agent"

$PY -m jtagent.io_snapshot                 # yearly snapshot (≈2 min)
$PY run_backtest.py 2018 2025              # rolling baseline backtest (8 years)
$PY trends.py                              # (a)/(b) signature trends
$PY -c "from jtagent import firm_panel; firm_panel.run()"   # (b) firm–year panel + job-split events
$PY -c "from jtagent import rule_c; rule_c.run_rule_c()"    # (c) template residuals and cross-firm adoption
$PY merge_anno.py 2019                     # merge LLM annotations
$PY rules_variants.py 2019 llm             # rule-family tournament (LLM / lex comparable on the same subset)
$PY run_agent_2019.py 2019 llm covered     # three-way clash + sample agent predictions
```

## 9. Artifact index

| File | Content |
|---|---|
| `out/results/baseline_metrics.csv` | full rolling baseline table for 2018–2025 |
| `out/results/signature_trend.csv` | yearly signature trends of (a)/(b) |
| `out/results/firm_year_stats.csv` | firm–year span panel |
| `out/results/split_events.csv` | job-split events (872 events) |
| `out/results/rule_c_trend.csv` | (c) template adoption and surface residuals |
| `out/results/agent_metrics_2019.csv` | pairwise backtest (LLM vs lexicon vs baselines) |
| `out/results/clash_2019.csv` | three-way clash (core table) |
| `out/results/agent_preds_2019_sample.parquet` | sample of the agent's high-score predictions (1,500 pairs, with mechanism labels and rationales) |
| `out/pair_judge_llm.json` | blind-judgment results for the 48 pairs |
