# Workflow in detail: the three-criteria "antecedent → outcome" conditional-probability test and repair

This document explains what every step of the entire pipeline does, what its inputs and outputs are, and how the scripts connect.
The scripts are in the `scripts/` directory; the data and judgment inputs used by the first-round judgment are in the `data/` directory.

> Status note: §0–§7 describe the original protocol and process of the first-round judgment (judge_input/judge_output were generated accordingly).
> In the first-round results, criterion b came out inverse and criterion c vacuous; the repair and cross-check of both are covered in §9.

---

## 0. Research design overview

Core idea: each of the three re-bundling criteria yields a testable prediction of the form "antecedent → T+h outcome".
For each unit of analysis, first use the LLM to judge, object by object, whether the antecedent holds (holds=0/1) and to give a probability bin p,
then use the measured outcome labels of years T+1/2/3 to compute the conditional-probability summary:

- P(outcome | holds=1) — the share of objects with the antecedent holding in which the outcome appears
- P(outcome | holds=0) — the share of objects with the antecedent not holding in which the outcome appears
- lift ratio = P(outcome|holds=1) / P(outcome|holds=0)

If the lift ratio > 1 and the holds=1 group is significantly higher than the holds=0 group, the criterion is supported by the data.

Units of analysis and alignment keys of the three criteria:

| Criterion | Unit | Alignment key | Outcome label field |
|------|------|--------|-------------|
| (a) complementarity re-bundling | job task set | `job` | `mixed_new` |
| (b) hierarchical unbundling | task cluster | `seed` | `split` |
| (c) mimetic template diffusion | firm | `firm` | `above_median` |

---

## 1. Data prerequisites

The pipeline depends on the following data, all under the `data/` directory of this repository:

- **Task dictionary** `data/snapshots/task_dict.parquet`: each task's text description, knowledge intensity E (E_ord 1–5), codability (codable), context-embeddedness (ctx), hierarchy (hierarchy), functional module (module)
- **Job snapshots** `data/snapshots/jobs_YYYY.parquet` (2016–2026): each job's task list per year (job → tasks)
- **Task-pair panel** `data/pairs/pairs_YYYY.parquet` (2018–2025): task-pair features and binding labels
- **Firm panel** `data/results/firm_year_jobs.parquet`: each firm's job list per year
- **Template similarities** `data/results/rule_c_adoption.parquet`: each job's struct_sim / xfirm_sim / surf_sim

These data are loaded through the interfaces of the `scripts/jtagent/` package (`P.load_jobs`, `P.load_edges`, `P.build_projection`, `firm_panel.build_firm_year_jobs`, etc.).

---

## 2. Step 1: object-pool construction — `scripts/objects.py`

### What it does

Build the judgment object pool for each of the three criteria separately: each object is the smallest unit for the subsequent LLM judgment and for outcome-label alignment.

### Criterion 1 g1: job task sets

- Source: the task lists of all jobs in a given year (≥2 tasks)
- Each object records: `job` (id), `tasks` (list of task ids), `e_ord` (each task's exposure level), `cod` (codability), `ctx` (context-embeddedness), `mod` (functional module), `has_E2`/`has_E0` (initial-judgment flags)
- Sampling: 2000 jobs drawn at random

### Criterion 2 g2: task clusters

- Source: on the task projection layer (the weighted projection of the co-occurrence network), take each task as a seed, take the strongest neighborhood within its own community (comm) by weight (top 8, ranked by job degree), and form tight task clusters of 3-15 tasks
- Each object records: `seed` (core task id), `comm` (community id), `tasks`, `e_ord`, `cod`, `hier` (hierarchy), `mod`, `e2_share` (share of E2 tasks), `hier_n` (number of hierarchy levels)
- After deduplication, 2000 task clusters sampled at random (actually about 1502)

### Criterion 3 g3: firms

- Source: the firm × job structure of a given year in the firm panel
- Each object records: `firm` (firm id), `n_jobs` (number of jobs), `mean_size` (average task-set size), `n_mod` (number of functional modules), `e2_share` (E2 share), `struct_sim` (mean structural similarity to the firm's own past jobs), `xfirm_sim` (mean cross-firm structural similarity), `surf_sim` (mean surface text similarity)
- Template residual = `surf_sim - max(struct_sim, xfirm_sim)`; the larger the residual, the more the wording looks like a template while the structure has not kept up
- In actuality about 307 firms

### Output

- `out/objects/g1_jobs_{year}.parquet`
- `out/objects/g2_comm_{year}.parquet`
- `out/objects/g3_firms_{year}.parquet`

---

## 3. Step 2: outcome labels — `scripts/outcomes.py`

### What it does

Compute the measured outcome labels at T+h (h=1/2/3 years) for each object.
The alignment keys are the same as the object pool's: g1=job, g2=seed, g3=firm.

### Criterion 1 outcome: `mixed_new`

- Definition: the job sees a new task combination in year T+h, and the new combination contains both high-exposure and low-exposure tasks
- How judged: compare the task sets of year T and year T+h; a job whose added tasks include both E2 and E0 is labeled 1

### Criterion 2 outcome: `split`

- Definition: the task cluster is carried in year T+h by ≥2 narrower jobs separately, and the carrying jobs' union coverage of the cluster's tasks is ≥60% (implementation in `outcomes.py`: holders = T+h jobs with task overlap ≥2 and a task count smaller than the cluster's size)
- Added after the repair, the directional outcome `down_std`: on top of split, require that the mean of the E_ord standard deviations of the carrying jobs' task sets be lower than the original cluster's — the new jobs split out are "purer" than the original cluster. See `scripts/verify_g2_direction.py` for the implementation

### Criterion 3 outcome: `above_median`

- Definition: among the firm's newly added jobs in year T+h, the template adoption share is higher than the same-year full-network median
- How judged: compute the template adoption share among newly added jobs firm by firm, then binarize at the same-year median

### Output

- `out/objects/labels_g1_{year}.parquet` (three rows per object, for h=1/2/3)
- `out/objects/labels_g2_{year}.parquet`
- `out/objects/labels_g3_{year}.parquet`

---

## 4. Step 3: judgment-input generation — `scripts/prep_judge_input.py`

### What it does

Render the object pool into LLM-readable batch files: the header of each file is the judgment protocol (antecedent definition + output format), and the body is one block per object (each object's task list, exposure levels, codability, context-embeddedness, and other fields).

### Judgment protocol (PROTO)

Each criterion's protocol text defines the judgment standard for whether the antecedent holds:

**Criterion 1**: the job task set simultaneously contains ① high-exposure codable tasks (E2, high codable) and ② context-embedded tasks (dependent on judgment/coordination/accountability/embodiment, low codable, E0/E1).
Both types present → holds=1; only one type, or both weak → holds=0.

**Criterion 2**: the tasks in the cluster satisfy ① same hierarchy level (hierarchy highly concentrated) ② predominantly codable (most tasks have explicit rules) ③ functionally cohesive (modules concentrated, homogeneous in task nature).
The three points basically satisfied → holds=1; hierarchy levels mixed, or a large share of context-embedded tasks → holds=0.

**Criterion 3**: the firm's job structure exhibits a "template residual" — surf_sim clearly higher than struct_sim and xfirm_sim, i.e., the wording resembles the prevailing templates but the task structure has not genuinely become isomorphic.
The larger the residual → the more the judgment leans toward holds=1; the three values close → holds=0.

### Batch splitting

- At most 60 objects per batch; a tail batch with fewer than 60 forms a file of its own
- g1 (2000 objects) → 34 batches, g2 (1502 objects) → 26 batches, g3 (307 objects) → 6 batches, 66 batches in total

### Output

- `out/judge_input/g1_batch_000.txt` … `g1_batch_033.txt`
- `out/judge_input/g2_batch_000.txt` … `g2_batch_025.txt`
- `out/judge_input/g3_batch_000.txt` … `g3_batch_005.txt`

Format of each file (the batch files on disk use Chinese field labels, shown verbatim):
```
【判定协议 · 准则(a) 互补性再捆绑 —— 前因判定】
...协议文本...
【输出】严格 JSON 数组...
[id=J38018 | 岗位任务集 5 任务 | 功能模块数 4]
  t9696 | Develop sales plans... | E2高 | 可编码0.50 语境0.25 | 销售
  ...
[id=J42299 | ...]
  ...
```

---

## 5. Step 4: LLM antecedent judgment — dynamic workflow

### What it does

Use the ZCode dynamic workflow to dispatch 66 sub-agents in parallel, each sub-agent responsible for judging one batch: read the batch file → judge object by object per the protocol → write JSON to the output directory.

### Workflow script

Script file: `scripts/workflow/pipeline.dwf.ts` (original path `.zcode/workflow-drafts/三准则前因判定.dwf.ts`)

### Execution logic

**Stage 1: parallel judgment (66 batches)**

- Create one sub-agent per batch (`判定员-{batch}`); the system prompt is uniformly set to "antecedent judger of the organizational-economics prediction agent", requiring it to judge strictly per the protocol, not fabricate ids, and say so truthfully if a file cannot be read
- Each sub-agent's task:
  1. Use the Read tool to read the corresponding batch file
  2. Judge, object by object, the three fields per the protocol:
     - `holds`: whether the antecedent holds, 0 or 1
     - `p`: probability bin, taking only one of 0.05/0.1/0.2/0.3/0.5/0.7/0.9
     - `why`: a Chinese-language rationale of ≤30 characters
  3. Use the Write tool to write the strict JSON array to `out/judge_output/{batch}.json`
  4. Reply with a single JSON object: `{"written": N, "first": "...", "last": "..."}`
- All batches are launched simultaneously via `Promise.all`; failed batches are each handled with try/catch inside the callback (a single batch's failure does not interrupt the entire workflow)

**Stage 2: fallback relabeling of failed batches**

- Collect the batches with status ≠ "ok" from Stage 1
- For each failed batch, create a fallback-relabeling sub-agent (`补标员-{batch}`) that re-executes the same judgment task
- The relabeling prompt says "the previous judgment was not successfully written to disk; please execute the whole task once more"
- Relabeling succeeds → counted into okCount, status marked as "ok-retry"

**Stage 3: summarize the write-to-disk manifest**

- Tally the count of successful batches and the list of batches that still failed
- Generate a markdown summary (artifact) listing the write-to-disk status

### Output

- `out/judge_output/g1_batch_000.json` … `g1_batch_033.json`
- `out/judge_output/g2_batch_000.json` … `g2_batch_025.json`
- `out/judge_output/g3_batch_000.json` … `g3_batch_005.json`

Format of each JSON file (the `why` field holds a Chinese rationale, shown verbatim):
```json
[
  {"id": "J38018", "holds": 1, "p": 0.3, "why": "含E2可编码与E0语境任务并存"},
  {"id": "J42299", "holds": 0, "p": 0.05, "why": "任务均可编码且同模块"},
  ...
]
```

### Failure handling

- A sub-agent throws an exception, or written=0 → marked failed, enters the fallback-relabeling stage
- Relabeling still fails → listed in the summary, needs manual handling
- The workflow itself does not fail as a whole because of a single batch's failure

---

## 6. Step 5: conditional-probability merging — `scripts/merge_judge.py`

### What it does

Merge the LLM judgment results (holds + p) with the outcome labels (measured at T+h), and output the main "antecedent → outcome" conditional-probability table and the probability-bin calibration table.

### Merging logic

- Inner join on the alignment key: g1 uses `job`, g2 uses `seed`, g3 uses `firm`
- Deduplicate the judgment results: for the same id, keep only the record of its first occurrence
- Judgment records whose id is not in the object pool are excluded

### Main conditional-probability table

Computed groupwise for each h (1/2/3):

| Column | Meaning |
|----|------|
| `n` | number of objects after alignment |
| `antecedent hold rate` | share of objects with holds=1 |
| `P(outcome\|holds=1)` | share of outcomes appearing in the holds=1 group |
| `P(outcome\|holds=0)` | share of outcomes appearing in the holds=0 group |
| `base rate` | overall share of outcomes across all objects |
| `lift ratio(holds=1/0)` | P(outcome\|holds=1) / P(outcome\|holds=0) |
| `n1`/`n0` | sample sizes of the two groups respectively |

### Probability-bin calibration table

Group by each h × p bin (0.05/0.1/0.2/0.3/0.5/0.7/0.9) and compute the actual outcome rate (mean) and the sample size (count) within that bin.
Used to check whether the probability bins given by the LLM correspond monotonically to the measured outcome rate —
the higher the p bin, the higher the actual outcome rate should be; otherwise this indicates calibration bias in the LLM.

### Output

- `out/objects/judge_g1_2020.parquet` (per-object judgments stored)
- `out/objects/judge_g2_2020.parquet`
- `out/objects/judge_g3_2020.parquet`
- `out/results/antecedent_outcome_2020.md` (conditional-probability summary table, markdown format)

---

## 7. Pipeline connection diagram (first round)

```
objects.py          →  g1_jobs / g2_comm / g3_firms parquet
      ↓
outcomes.py         →  labels_g1 / g2 / g3 parquet (T+1/2/3 outcome labels)
      ↓
prep_judge_input.py →  judge_input/*.txt (66 batches, with judgment protocol)
      ↓
workflow/pipeline.dwf.ts →  judge_output/*.json (66 batches of LLM judgments)
      ↓
merge_judge.py      →  judge_g* parquet + antecedent_outcome_2020.md
```

First-round conclusions: a holds (2.38/3.00/2.57); b inverse (0.57/0.56/0.49); c vacuous. Go to §9 for the repair.

---

## 8. Reproduction notes

1. All scripts hard-code the path `D:\harness office\research\jobtask-agent\`; migrating to another environment requires modifying path variables such as `BASE` / `OUT` / `OBJ`
2. `objects.py` depends on the `scripts/jtagent/` package (the `pairs`, `attrs`, `firm_panel` modules)
3. The LLM judgment workflow depends on ZCode's dynamic-workflow feature and must run in the ZCode environment
4. The outcome labels depend on task-network snapshots of historical years (building the T+h comparison requires data from after year T)
5. The probability bin p has only 7 levels, so precision is limited; if finer calibration is needed, extend the bins or switch to continuous values

---

## 9. Repair and cross-check phase (after the first round)

The first round exposed two problems: b inverse and c vacuous. The repair was carried out in four steps — "diagnosis → redefinition → verification → independent cross-check"; all scripts are committed to the repository, and the numbers of every step have a corresponding results document.

### 9.1 Diagnosis (why inverse/vacuous)

- `scripts/reverse_outcome.py`: inverse-probability table — when the outcome occurs, is the antecedent more likely to hold
- `scripts/diagnose_g2.py`: attribution of b's inverse relationship. Compare cluster features by holds group: clusters with holds=1 average 4.99 tasks, with hier_n mean 1.92; clusters with holds=0 have 6.38 tasks, hier_n 2.65. "The judgment holds" filters out smaller clusters that are harder to split as a whole — the inverse relationship comes from the confounding structure rather than a mechanism error
- `scripts/diagnose_g3.py`: confirmation that c is vacuous. The hold rate is 82.5%, the p bins have no monotonic relationship with the measured outcomes, and tightening the judgment standard cannot recover the discrimination either

### 9.2 Criterion b: threshold scan + directional reconstruction

**Threshold scan** (`scripts/scan_g2_threshold.py` → `docs/results/g2_threshold_scan.md`):
scan univariate and joint splits over the four candidate thresholds size / hier_n / n_mod / e2_share; hier_n≥3 is optimal — under this threshold the lift ratio jumps from 0.55 to 1.68 and the inverse relationship disappears.

**Threshold-version verification** (`scripts/verify_g2_fix.py` → `docs/results/g2_fix_verify.md`):
the new antecedent holds∧hier_n≥3 (n1=49) has lift ratios 2.34/2.59/2.49 and inverse lift ratios 5.67/4.78/3.80, chi-square p<7.3e-07 (three windows).

**Directional reconstruction** (`scripts/verify_g2_direction.py` → `docs/results/g2_direction_verify.md`):
following Garicano's original meaning, change the outcome from "whether it splits" to "in which direction it splits":

- Antecedent asym: the standard deviation of the E_ord of the tasks within the cluster (knowledge asymmetry, median split) — does not depend on the LLM
- Outcome down_std: split, and the mean of the E_ord standard deviations of the carrying jobs' task sets is lower than the original cluster's
- Result: lift ratios 1.56/1.40/1.29, significant across all three windows (p=4.6e-06/0.003/0.038)
- The original inverse puzzle is thereby solved: high-asymmetry clusters are larger and more complex and are, on the contrary, less likely to be split as a whole (0.77/0.74/0.67 under the original split criterion), but the direction they split out in matches the theory
- Incidental finding: the down_e lift ratio is 0.25 (strongly inverse): the narrow jobs split out have higher average knowledge intensity — the actual mechanism is knowledge-intensive tasks peeled off into specialist jobs (the knowledge layer peeling away from the execution layer), not "routine tasks sinking down"
- Stacking the criterion holds∧hier_n≥3∧asym_high: lift ratios 3.36/2.74/2.73 (n=15, upper-bound evidence)

### 9.2bis Job-version (the primary test of criterion b)

The theoretical statement of criterion b takes the **job** as its unit: a job task set with high knowledge asymmetry → that task set congeals into new jobs of lower asymmetry. The first-round object pool used task clusters (task co-occurrence communities) to carry the "task set", which does not correspond directly to the theory. `scripts/verify_g2_job.py` (→ `docs/results/g2_job_verify.md`) transplants the same antecedent/outcome onto job task sets (g1 object pool of 2000 jobs, asym = the standard deviation of the tasks' E_ord within the job, median split; the outcome-matching logic is the same as label_g2: narrow jobs in year T+h that carry a subset of its tasks):

| Outcome | h=1 | h=2 | h=3 | Significance |
|------|-----|-----|-----|--------|
| down_std (directional) | **2.65** | **2.86** | **2.95** | p=1.3e-18/7.4e-14/3.7e-12 |
| split (control, no direction constraint) | 1.14 | 1.15 | 1.13 | all non-significant |
| down_e (narrow jobs' mean E lower) | 0.46 | 0.49 | 0.42 | strongly inverse |

The job-version's strength is about 2 times that of the task-cluster version and increases with h; the neutral split control confirms that "old jobs do not disappear that fast, and it is the split direction that Criterion 2 predicts". The task-cluster version (the directional reconstruction of §9.2) is retained as a variant. Robust in the size≥4 subsample (2.20/2.37/2.45).
New-job-restricted robustness (`scripts/verify_g2_job_newonly.py`): after restricting the carrying jobs to jobs newly entering in year T+h, the lift ratios are 2.55/2.97/2.95 (p≤9.5e-17); the signal is driven by new jobs.

### 9.3 Criterion c: objective-antecedent replacement

**Metric scan** (`scripts/scan_g3_metrics.py` → `docs/results/g3_metric_scan.md`):
median splits and Spearman rank-power tests on 9 objective structural metrics, compared against the LLM holds baseline:

| Metric | h=1 | h=2 | h=3 | Significance |
|------|-----|-----|-----|--------|
| n_jobs | 2.57 | 2.51 | 1.90 | strongly significant at h=1/2 |
| n_mod | 2.06 | 2.49 | 2.70 | significant in all three windows |
| surf_sim | 0.80 | 0.95 | 0.62 | inverse throughout |
| struct_sim | 1.60 | 1.07 | 1.19 | no stable signal |
| LLM holds (control) | 1.27 | 1.72 | 0.98 | all non-significant |

**Fix verification** (`scripts/verify_g3_fix.py` → `docs/results/g3_fix_verify.md`):
the antecedents are median splits of n_jobs/n_mod; the combined antecedent n_jobs high∧n_mod high has lift ratios 2.98/3.26/3.06 (p=8.0e-06/2.6e-05/0.0015), inverse lift ratios 1.93/1.98/1.74. Note: the continuous mean of adopt_share is zero-inflated at the firm level (small denominators); direction follows the binary label above_median.

### 9.4 Independent cross-check

`scripts/recheck_results.py` / `scripts/recheck_g2fix.py`: the 18 key tests are independently recomputed with `scipy.stats.chi2_contingency` (correction=False), agreeing to the last digit with the hand-written two-proportion z / 2×2 chi-square formulas; sample sizes and NaN checks are verified as well.

### 9.5 Connection diagram after the repair

```
diagnose_g2.py / diagnose_g3.py          (pinpointing: confounding structure / vacuous)
      ↓
scan_g2_threshold.py → verify_g2_fix.py          b threshold version
      ↓
verify_g2_direction.py                           b directional version (antecedent made objective)
      ↓
scan_g3_metrics.py → verify_g3_fix.py            c objective-antecedent version
      ↓
recheck_results.py / recheck_g2fix.py            scipy independent cross-check
      ↓
all three criteria hold (the README "core results" table)
```
