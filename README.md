# job-behind-name

> **English translation.** This repository is the English translation of [job-behind-name](https://github.com/very11ok/job-behind-name) (Chinese original). Where wording could be ambiguous, the Chinese original is authoritative; all numbers, code identifiers, and file paths are identical between the two.

A job–task network derived from job-posting text, used to implement three "re-bundling" criteria from organization theory and labor economics as testable prediction rules: the network structure at **T=2020** serves as the antecedent, the job–task combinations that actually emerged in **T+1/2/3** (2021–2023) serve as the outcome, and the test is a conditional-probability judgment of "antecedent → several years later → outcome". The LLM is used for the subjective judgment of antecedents and for job–task pair binding judgments; all statistics (conditional probabilities, lift ratios, chi-square, Spearman) are computed deterministically. The repository contains all data, scripts, intermediate artifacts, and result documents, and is fully reproducible.

## The Three Criteria

Each criterion is described in four layers: what it is, the phenomenon it predicts, how it is validated (baseline experiment and hardening), and what you can do next.

| # | Criterion | Theory | Unit of analysis | Antecedent (T=2020) | Outcome (T+h) |
|---|------|----------|----------|----------------|-------------|
| 1 | Complementarity re-bundling | Milgrom & Roberts 1995 / Kremer 1993 | job task set | High-exposure codable tasks coexist with context-embedded tasks (LLM judgment) | Mixed new combinations (mixed_new) |
| 2 | Hierarchical unbundling | Garicano 2000 | job task set | High knowledge asymmetry (dispersion of task E_ord) | Narrower jobs with lower asymmetry congeal (down_std) |
| 3 | Mimetic template diffusion | DiMaggio & Powell 1983 | firm × year | n_jobs, n_mod above the median | Template adoption share above the same-year median (above_median) |

### Criterion 1: Complementarity Re-bundling

**What it is.** The Milgrom–Roberts complementarity proposition and Kremer's O-ring theory say: productivity depends on how well tasks match each other. AI strengthens the complementarity between high-exposure codable tasks (E2: data analysis, report generation) and context-embedded tasks (E0/E1: relying on judgment, coordination, accountability) — the payoff of bundling both kinds of tasks into the same job rises. Criterion 1 therefore predicts: AI does not eliminate jobs; it changes which tasks a job packs.

**The predicted phenomenon.** When the task set of a job that satisfies the antecedent changes at T+h, the newly appearing task pairs mix in both high- and low-knowledge-intensity tasks — the job grows "mixed new combinations". The time trend in the data points the same way: from 2017 to 2025, the average number of tasks per new job rose 4.90→6.26, and the mixed high/low-exposure share rose 0.575→0.658 — jobs are getting broader and more mixed.

**Baseline experiment.** At T=2020, 2000 jobs are sampled (alignment key `job`); the LLM judges, job by job under the protocol, whether the antecedent holds (high-exposure codable tasks and context-embedded tasks coexist within the task set); the outcome mixed_new is computed from the actual 2021–2023 snapshots. The conditional-probability lift ratio is **2.38 / 3.00 / 2.57** (h=1/2/3), significant in all three windows; the inverse probabilities point the same way; the antecedent holds for 43.75% of jobs, so the judgment has discriminating power. This is the only one of the three criteria that passes with its original judgment form.

**Hardening.** Spot checks of judgment quality confirm the judgments are not misclassifications; 18 key statistics were independently re-verified with scipy and agree exactly. An objectified reformulation (aligned with how Criterion 2 was handled), multi-start tests, and refining the binary outcome into a continuous measure are left as future extensions.

**What you can do next.** Given a firm's 2020 job list, computing the antecedent job by job yields a priority list for job re-bundling — jobs in the antecedent-holds group are 2.4–3× more likely to develop mixed new combinations within two to three years than jobs in the does-not-hold group. Research questions: an objectified reformulation of the antecedent; multi-start tests moving T back to 2018/2019; refining the binary outcome into a continuous mixedness value.

### Criterion 2: Hierarchical Unbundling

**What it is.** Garicano's knowledge-based hierarchy theory says: a job bundles knowledge-intensive tasks and routine tasks onto one person because the tacit knowledge of "which knowledge to invoke when an exception comes up" lives only in that person's head — hierarchies exist because knowledge is unevenly distributed. Once AI encodes the knowledge-intensive part into systems, this vertical bundling comes apart: knowledge-intensive tasks are peeled off the original job and become independent narrow jobs.

**The predicted phenomenon.** (1) Jobs with high knowledge asymmetry congeal, out of their task sets, narrow jobs with lower asymmetry, and the signal strengthens the longer the observation window; (2) looking only at "whether it splits at all" (the split control) has no discriminating power — old jobs do not disappear quickly, and the predictive content lies entirely in "which direction it splits"; (3) the narrow jobs that split out are on average higher in knowledge intensity — the actual mechanism is knowledge-intensive tasks being peeled off into specialist jobs, not routine tasks sinking down to lower-skill workers.

**Baseline experiment.** The unit of analysis is the job; 2000 jobs. Antecedent asym = the standard deviation of task E_ord within the job (median split; an objective quantity, no LLM involved); outcome down_std = at T+h, ≥2 narrow jobs emerge carrying subsets of its tasks, with asymmetry lower than the original job. The conditional-probability lift ratio is **2.65 / 2.86 / 2.95** (p=1.3e-18 / 7.4e-14 / 3.7e-12), increasing with the observation window — the strongest test in the whole project.

**Hardening.** Four robustness checks: (1) the split control (direction constraint removed) gives 1.13–1.15, all non-significant — "whether it splits" has no discriminating power, the predictive content is all in the direction; (2) the subsample with task sets ≥4 gives 2.20/2.37/2.45; (3) the task-cluster variant gives 1.56/1.40/1.29 (the task co-occurrence community serves as a networked definition of the "task set", and the conclusion points the same way); (4) restricting the carrying jobs to newly entering jobs gives 2.55/2.97/2.95 — the splitting signal is driven by new jobs. The enhanced version stacking the LLM semantic judgment on top gives 3.36/2.74/2.73 (n=15). The repair path of the judgment form (the original LLM judgment ran in reverse at 0.57/0.56/0.49; then confounding-structure diagnosis, threshold hier_n≥3, directional reformulation, and unit correction to the job) is documented in full in the "Project History" section and `docs/workflow.md` §9.

**What you can do next.** It works directly as a job-level predictor: feed in any job's task set, compute asym, and you get the probability that it splits out purer new jobs within two to three years (13–23% in the antecedent group vs 4–9% in the control group). Research questions: what form do the split-off specialist jobs take (the share of AI-calibration/supervision-type jobs); how the speed of knowledge peeling-off relates to AI exposure; differences in splitting rates between craft-intensive and information-intensive industries.

### Criterion 3: Mimetic Template Diffusion

**What it is.** DiMaggio–Powell's institutional isomorphism theory says: firms in an organizational field face mimetic pressure, and job structures replicate prevailing templates at the cross-firm level.

**The predicted phenomenon.** Among the T+h newly added jobs of larger firms (more jobs, more functional modules), the cross-firm template adoption share is above the same-year median — large, multi-module organizations move first, in the direction institutional isomorphism theory expects.

**Baseline experiment.** 307 firms (alignment key `firm`); antecedent = median split of organizational size-structure indicators (n_jobs high ∧ n_mod high, objective quantities); outcome above_median = the template adoption share among new jobs is above the same-year median. The conditional-probability lift ratio is **2.98 / 3.26 / 3.06** (p=8.0e-06 / 2.6e-05 / 0.0015), significant in all three windows, with the effect size flat across the three observation windows (2.98→3.06); inverse lift ratio 1.93/1.98/1.74.

**Hardening.** How the antecedent form was settled: the original LLM judgment (template-imprint residual) was diagnosed and confirmed to be vacuous (82.5% of firms judged holds=1, and the p buckets had zero ranking power against the measured outcome), so it was dropped; 9 objective structural indicators were scanned one by one, and n_jobs and n_mod won (univariate: n_mod 2.70/2.88/3.23, n_jobs 2.59/2.38/2.08; the combination is strongest), while all similarity indicators were ineffective or reversed. A zero-inflation footnote on the continuous indicator: small firms have small adopt_share denominators, so direction is judged by the binary label.

**What you can do next.** Predict which firms will become template adopters — large, multi-module organizations move first, and institutional theory's expectation gets a quantified lift ratio here. Research questions: whether the adopted template jobs have higher AI exposure (the link between isomorphism and AI diffusion); the time trend of industry-level "templatization rates"; the within-firm ordering between template adoption and Criterion 1's mixed re-bundling.

### The Common Validation Framework

One common protocol: T=2020 antecedent → measured outcomes at T+1/2/3 → conditional probabilities P(outcome|antecedent=1) vs P(outcome|antecedent=0) → lift ratio + two-proportion z test + 2×2 chi-square + inverse probability → 18 sets of key numbers independently re-verified with `scipy.stats.chi2_contingency`, agreeing exactly. The alignment keys job / seed / firm run through the object pools, the outcome labels, and the judgment results. The LLM only performs semantic judgment (task-pair binding, the semantic layer of the antecedents); all statistical conclusions are produced by deterministic code. Known boundary: T=2020 is the single starting point; the multi-start tests (2018/2019) have the framework and data ready but have not been run.

## Core Results (lift ratio = P(outcome|antecedent=1) / P(outcome|antecedent=0), T=2020)

| Criterion / test | h=1 | h=2 | h=3 | Significance | Verdict |
|-------------|-----|-----|-----|--------|------|
| 1: LLM holds → mixed_new (original form) | **2.38** | **3.00** | **2.57** | significant in all three windows | passes natively |
| **2: job-version high asym → down_std (main test)** | **2.65** | **2.86** | **2.95** | p≤7.4e-14 | **passes (strongest in the project)** |
| 2: task-cluster-version high asym → down_std (variant) | 1.56 | 1.40 | 1.29 | p=4.6e-06/0.003/0.038 | passes |
| 2: LLM holds ∧ hier_n≥3 → split (threshold version, variant) | **2.34** | **2.59** | **2.49** | p<7.3e-07 | passes |
| 2: threshold version ∧ high asym → down_std (stacked) | **3.36** | **2.74** | **2.73** | p≤0.006 | passes (n=15) |
| 3: n_jobs high ∧ n_mod high → above_median | **2.98** | **3.26** | **3.06** | p=8e-06/2.6e-05/0.0015 | passes |
| 3: LLM holds (control; confirmed vacuous) | 1.27 | 1.72 | 0.98 | all non-significant | dropped |

(Criteria 1/2/3 correspond to g1/g2/g3 in scripts and data files.)

## The Pipeline at a Glance

```
Data      data/snapshots/ (2016–2026 job–task snapshots + task_dict task dictionary)
   │      jtagent/pairs.py → data/pairs/ task-pair panel (common neighbors / preferential attachment / Jaccard / knowledge gap and other features)
   │      jtagent/attrs.py → task semantic attributes (knowledge intensity E, codability, context embedding, hierarchy, functional modules)
   ▼
Objects   scripts/objects.py     object pools for the three criteria, data/objects/g{1,2,3}_2020.parquet
   │                             g1 2000 job task sets / g2 1502 task clusters / g3 307 firms
   ▼
Outcomes  scripts/outcomes.py    measured outcome labels at T+h (h=1/2/3), labels_g*_2020.parquet
   │                             per object, look back at the actual 2021–2023 network and ask "what happened next"
   ▼
Judgment  scripts/prep_judge_input.py → data/judge_input/ (66 batches, PROTO protocol + object details)
   │      scripts/workflow/pipeline.dwf.ts → 66 subagents run LLM judgments in parallel (with fallback relabeling)
   │      scripts/merge_judge.py → judge_g*_2020.parquet + main conditional-probability table
   ▼
Diagnosis scripts/reverse_outcome.py / diagnose_g2.py / diagnose_g3.py
   │      Round-1 conclusions: 1 passes positively; 2 runs in reverse (0.57/0.56/0.49); 3 vacuous (82.5% judged 1, p buckets no ranking power)
   ▼
Repair    scripts/scan_g2_threshold.py → verify_g2_fix.py     2: threshold hier_n≥3
   │      scripts/verify_g2_direction.py                       2: directional asym → down_std (task-cluster version)
   │      scripts/verify_g2_job.py                             2: job version (final form)
   │      scripts/scan_g3_metrics.py → verify_g3_fix.py        3: objective antecedents n_jobs/n_mod
   ▼
Recheck   scripts/recheck_results.py / recheck_g2fix.py       independent cross-validation with scipy.stats
```

The alignment keys `job` / `seed` / `firm` stay consistent across the three layers: object pools, outcome labels, and judgment results.

## Project History: From One Criterion Passing to All Three Passing

**Round 1 (original judgment forms)**: after the 66 batches of LLM antecedent judgments completed, only one of the three criteria passed. Criterion 1: lift ratio 2.38/3.00/2.57, significant in all three windows. Criterion 2 ran in reverse — clusters the LLM judged as "hierarchical unbundling holds" split at a lower rate (0.57/0.56/0.49). Criterion 3 was weakly positive but all non-significant (1.27/1.72/0.98), and the LLM judged holds=1 for 82.5% of firms, with no monotone relation between the p buckets and the measured outcomes.

**Diagnosis**: `diagnose_g2.py` localized Criterion 2's reversed relation to a confounding structure — clusters with holds=1 are smaller on average (4.99 vs 6.38 tasks), with fewer modules and fewer hierarchy levels; such clusters are inherently hard to split, so "the LLM says holds" was selecting exactly the hard-to-split clusters. `diagnose_g3.py` confirmed Criterion 3's judgment was vacuous: the holds rate was too high, and the confidence bins were decoupled from the outcomes — the LLM produced no usable information on this object type.

**Repair**: three lines of work completed in parallel. (1) Criterion 2's threshold scan found hier_n≥3 as the best split among the four candidates size/hier_n/n_mod/e2_share; the new antecedent holds∧hier_n≥3 flipped the lift ratio from 0.55 to 2.34/2.59/2.49 (chi-square p<7.3e-07). (2) Criterion 2's directional reformulation rewrote the outcome in Garicano's original sense: it no longer only required "splitting", but required the narrow jobs that split out to have task-set asymmetry lower than the original cluster (down_std); the antecedent became the standard deviation of task E_ord within the cluster, decoupled from the LLM. Lift ratio 1.56/1.40/1.29, significant in all three windows, and it also explains round 1's reversed relation — high-asymmetry clusters are larger and more complex, so the chance of being split as a whole is lower, but whenever a split happens, its direction agrees with the theory. Incidental finding: the narrow jobs that split out are higher in average knowledge intensity (task-cluster version down_e lift ratio 0.25) — knowledge-intensive tasks are peeled off into specialist jobs, rather than routine work sinking down. (3) Criterion 3 dropped the LLM judgment; the objective-indicator scan tested 9 structural indicators, n_jobs and n_mod won, all similarity indicators were ineffective, and the combined antecedent n_jobs high ∧ n_mod high gives 2.98/3.26/3.06.

**Cross-check**: all 18 sets of key tests (conditional probabilities, chi-square, inverse probabilities) were independently recomputed with `scipy.stats.chi2_contingency`, agreeing exactly with the hand-written formulas; `recheck_results.py` / `recheck_g2fix.py` are in the repository and can be re-run.

**Unit correction (Criterion 2's final form)**: Criterion 2's theoretical statement is "a job task set with high knowledge asymmetry → new jobs with lower asymmetry congeal under that task set", so the unit of analysis should be the **job**. The object pool inherited from round 1 used the task cluster (task co-occurrence community) as the carrier of the "task set", which does not correspond directly to the theory. `verify_g2_job.py` transplants the antecedent and outcome as-is onto job task sets (2000 jobs; asym = standard deviation of task E_ord within the job; outcome = narrow jobs with lower asymmetry emerging under the task set): lift ratio 2.65/2.86/2.95 (p=1.3e-18/7.4e-14/3.7e-12), roughly twice the task-cluster version and increasing over time; the split control is neutral (1.13–1.15, p>0.1). The narrow jobs have higher average E (down_e lift ratio 0.42–0.49) — knowledge-intensive tasks are peeled off into specialist jobs rather than routine work sinking down, consistent with the task-cluster version. New-job-only robustness (`verify_g2_job_newonly.py`): restricting the carrying jobs to jobs newly entering at T+h gives 2.55/2.97/2.95 (p≤9.5e-17), consistent with the all-jobs version — the signal is driven by new jobs. The task-cluster version is kept as a variant.

**Methodological positioning**: the LLM in this project performs semantic judgment (task-pair binding; the semantic layer of whether antecedents hold), while all statistical conclusions are produced by deterministic code. Where the LLM judgment conflicted with structural evidence, the diagnosis pipeline above localized the conflict to a specific confounding structure, and the antecedent was replaced with an objective quantity measurable independently.

## Repository Layout

```
├── README.md                        this file
├── docs/
│   ├── workflow.md                  9-section detailed description of the workflow and repair stages
│   ├── report.md                    2019 pair-by-pair backtest report (historical stage; status note at the top)
│   └── results/*.md                 all result documents
├── scripts/
│   ├── *.py                         main pipeline scripts (see "Script Index")
│   ├── jtagent/*.py                 core library: snapshot IO / similarity / attributes / LLM client / rules
│   └── workflow/pipeline.dwf.ts     dynamic-workflow script for the 66-batch LLM antecedent judgments
└── data/
    ├── snapshots/                   yearly task-network snapshots (2016–2026, 23 parquet files)
    ├── pairs/                       yearly task-pair similarity panels (2018–2025, 13 parquet files, ~376MB)
    ├── objects/                     object pools for the three criteria + outcome labels + LLM antecedent judgments (9 parquet files)
    ├── results/                     all result tables (csv/parquet)
    ├── judge_input/                 66 batches of LLM antecedent judgment inputs (PROTO protocol text)
    ├── anno_input_2019/             manual annotation materials for the 2019 binding judgments (120 batch csv files)
    ├── anno_output_2019/            2019 annotation collection and construction scripts
    └── pair_judge_truth.parquet     ground-truth set for the job–task pair binding judgments (manual annotation + LLM comparison)
```

## Data Dictionary

### data/snapshots/ (task-network snapshots)

- `jobs_YYYY.parquet` (YYYY=2016…2026): `job` job ID, `tasks` list of task IDs, `name` job name, `firm` firm, `industry` industry, `exposure` AI exposure, `n_postings` number of job postings.
- `edges_YYYY.parquet`: job–task edges (`job`, `task`).
- `task_dict.parquet`: task dictionary. `task` ID, `task_text` task text, `soc`/`occupation` occupation classification, `E`/`E_human` knowledge-intensity labels, `auto` automatability label, `E_ord` numeric knowledge intensity (1–5; the base for computing Criterion 2's asymmetry asym).

### data/pairs/ (task-pair similarity panels, 2018–2025)

`pairs_YYYY.parquet`: task pairs `ti, tj` and features — common-neighbor count `cn`, degrees `deg_*`, preferential attachment `pa`, Jaccard, knowledge intensity `E_ord_*`, context `ctx_*`, codability `codable_*`, module membership `module_*`, knowledge gap `E_gap`; `y`/`y_new` binding-continuation/new-binding labels; `bind_TM1` previous-year binding. Use: task-pair-level prediction base for Criteria 1/2. Recomputed from the snapshots by `scripts/jtagent/pairs.py`.

> File-splitting note: GitHub's per-file limit is about 40MB; 2019/2020/2023 are complete single files, while 2018/2021/2022/2024/2025 are split in half by rows into `_part1/_part2` (2.5 million rows each); reading them with `pd.concat([pd.read_parquet(p1), pd.read_parquet(p2)])` restores them losslessly. Splitting script: `scripts/split_pairs.py`.

### data/objects/ (object pools for the three criteria, T=2020)

- `g1_jobs_2020.parquet` (2000 jobs): `job`, `size` number of tasks, `tasks`/`e_ord`/`cod`/`ctx`/`mod` task-attribute strings, `has_E2`/`has_E0` high/low-knowledge task flags, `n_mod` number of modules. Alignment key `job`.
- `g2_comm_2020.parquet` (1502 task clusters): `seed` seed task, `comm` community, `size` cluster size, `tasks`, `e_ord`, `hier`, `mod`, `n_mod`, `e2_share` share of codable tasks, `hier_n` number of hierarchy levels. Alignment key `seed`.
- `g3_firms_2020.parquet` (307 firms): `firm`, `n_jobs` number of jobs, `mean_size`, `n_mod`, `e2_share`, `struct_sim`/`xfirm_sim`/`surf_sim` structural/cross-firm/surface similarity. Alignment key `firm`.
- `labels_g1_2020.parquet`: `job, h, alive, changed, n_new_pairs, n_mixed_new_pairs, mixed_new` (Criterion 1 outcome).
- `labels_g2_2020.parquet`: `seed, h, n_holders, cover, split` (Criterion 2 outcome: carried by ≥2 narrow jobs with coverage ≥60%).
- `labels_g3_2020.parquet`: `firm, h, n_new_jobs, adopt_share, adopt_mean_sim, above_median, median_adopt` (Criterion 3 outcome).
- `judge_g1/g2/g3_2020.parquet`: LLM antecedent judgments; `holds` (0/1), `p` confidence bin, `why` reason, `batch` batch number.

### data/results/ (result tables)

- Repair-verification details: `g2_threshold_scan` (threshold scan), `g2_threshold_combo` (combined threshold), `g2_fix_verify` (threshold-version verification), `g2_direction_verify` (directional reformulation / task-cluster variant), `g2_job_verify` (job-version main test), `g2_job_newonly` (new-job-only robustness), `g3_metric_scan`/`g3_metric_corr` (indicator scan), `g3_fix_verify` (Criterion 3 repair verification); each has a .md (under docs/results/) and a .parquet.
- `rule_c_adoption.csv/.parquet` (base material for Criterion 3): per new job, `struct_sim`, `xfirm_struct_sim`, `surf_sim`, matched templates `tmpl_*`, `cross_firm`, `surf_resid`.
- `split_events.csv` split events, `firm_year_jobs.parquet` and `firm_year_stats.csv` firm×year panels, `agent_metrics_2019.csv`, `baseline_metrics.csv`, `ablation_feats_*` ablations of the 2019 prediction experiment, `signature_trend.csv`, `snapshot_summary.csv`, `candidate_summary.csv`, and `anno_2019.parquet` with `agent_preds_2019_sample.parquet` (2019 annotation experiment).

### data/judge_input/ and data/pair_judge_truth.parquet

- `judge_input/*.txt`: 66 batches of LLM antecedent judgment inputs — PROTO judgment protocol + BATCH details — corresponding to the `batch` field of `judge_g*_2020.parquet`.
- `pair_judge_truth.parquet`: ground-truth set for the job–task pair binding judgments (2019 manual annotation + LLM comparison), with task text, features, and annotation columns such as `A_互补再捆绑`/`B_层级解绑`.

### data/anno_input_2019/ and data/anno_output_2019/

Manual annotation materials for the 2019 job–task pair binding judgments (120 batches of csv inputs + collected annotations and construction scripts); the source of `pair_judge_truth.parquet` and of the Criterion 1/2 calibration.

## Script Index (scripts/, in execution order)

1. `jtagent/io_snapshot.py` — snapshot loading; `jtagent/pairs.py` — task-pair panel construction
2. `jtagent/attrs.py` / `features.py` / `firm_panel.py` / `rule_c.py` — task attributes, features, firm panels, template adoption
3. `objects.py` — builds the object pools for the three criteria → `data/objects/g*_2020.parquet`
4. `outcomes.py` — generates the outcome labels → `labels_g*_2020.parquet`
5. `prep_judge_input.py` — generates the 66 batches of judgment inputs → `data/judge_input/`
6. `workflow/pipeline.dwf.ts` — the LLM judgment workflow (66-batch fan-out + fallback relabeling)
7. `merge_judge.py` — merges judgments with labels → `judge_g*_2020.parquet`
8. `reverse_outcome.py` / `diagnose_g2.py` / `diagnose_g3.py` — inverse probabilities and diagnosis
9. `scan_g2_threshold.py` / `verify_g2_fix.py` — Criterion 2 threshold scan and verification
10. `verify_g2_direction.py` — Criterion 2 directional-reformulation verification (task-cluster variant)
11. `verify_g2_job.py` — Criterion 2 job-version verification (main test: job task-set asym → down_std)
12. `scan_g3_metrics.py` / `verify_g3_fix.py` — Criterion 3 indicator scan and repair verification
13. `recheck_results.py` / `recheck_g2fix.py` — independent cross-checks with scipy
14. `run_agent_2019.py` / `run_backtest.py` / `ablation_feats.py` / `make_figures.py` / `trends.py` — the 2019 prediction experiment and figures
15. `check_hedging.py` — detects defensive writing (hedging) in documents; the lexicon comes from the Chinese hedge word list of [aproko/keyword_hedge_detector](https://github.com/aproko/keyword_hedge_detector) (colloquial words filtered out), supplemented with English hedges; used for wording QC before publishing documents

## Reproducing the Results

```bash
# 1. Object pools + outcome labels (T=2020)
python objects.py && python outcomes.py
# 2. Judgment inputs (66 batches) → run workflow/pipeline.dwf.ts with the workflow engine → merge
python prep_judge_input.py && python merge_judge.py
# 3. Repair verification for the three criteria
python verify_g2_fix.py && python verify_g2_direction.py && python verify_g2_job.py && python verify_g3_fix.py
# 4. Independent cross-checks (scipy cross-validation)
python recheck_results.py && python recheck_g2fix.py
```

Dependencies: `pandas pyarrow numpy scipy`. The LLM client (`jtagent/llm.py`) reads endpoint configuration from the environment variable `JTAGENT_ENDPOINTS` (JSON: `[{"baseURL":..., "apiKey":..., "name":...}]`); the repository contains no credentials.

## Results Document Index (docs/results/)

`antecedent_outcome_2020.md` (main table) · `reverse_outcome_antecedent_2020.md` (inverse probabilities) · `g2_threshold_scan.md` (threshold scan) · `g2_fix_verify.md` (threshold-version verification) · `g2_direction_verify.md` (directional reformulation / task-cluster variant) · `g2_job_verify.md` (job-version main test) · `g3_metric_scan.md` (indicator scan) · `g3_fix_verify.md` (Criterion 3 repair verification). The confounding-diagnosis report `diagnose_g2_g3.md` is at the root of `docs/`. The numeric results of the 2019 pair-by-pair backtest are in `data/results/` (csv/parquet); the narrative is in `docs/report.md`.
