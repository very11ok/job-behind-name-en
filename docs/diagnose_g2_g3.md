# g2/g3 diagnosis report: inverse relationship and vacuous shell confirmed

## g2 (Criterion b, hierarchical unbundling) — the inverse relationship is an artifact of confounding variables

### Phenomenon

The forward lift ratio is 0.49-0.57 and the inverse lift ratio is 0.51-0.57; both directions consistently show a negative association.

### Diagnosis 1: the holds=1 clusters and the split=1 clusters are opposite profiles

| Feature | Clusters with holds=1 (n=1227) | Clusters with holds=0 (n=3279) | Clusters with split=1 (n=1189) | Clusters with split=0 (n=3317) |
|------|--------------------:|--------------------:|--------------------:|--------------------:|
| Size (size) | 4.99 | 6.38 | 8.42 | 5.14 |
| Number of modules (n_mod) | 2.48 | 3.89 | 4.16 | 3.27 |
| Number of hierarchy levels (hier_n) | 1.92 | 2.65 | 2.80 | 2.33 |
| Same-layer share (e2_share) | 0.608 | 0.605 | 0.679 | 0.579 |

The LLM assigned "the antecedent holds" to **small, simple, low-hierarchy** clusters, but splits actually happen in **large, complex, high-hierarchy** clusters. The two have opposite structural profiles, which is why the raw data show an "inverse relationship".

### Diagnosis 2: after stratification the inverse relationship is not robust

After splitting into three groups by size or hier_n, the inverse relationship disappears in the middle and high groups and even turns positive:

| Stratification variable | Low-group lift ratio | Middle-group lift ratio | High-group lift ratio |
|----------|----------:|----------:|----------:|
| size | 0.00 | 1.57 | 1.04 |
| hier_n | 0.52 | 1.79 | 1.96 |

The inverse relationship exists only in the low group; in the middle and high groups the lift ratio is > 1.

### Diagnosis 3: logistic regression settles it

After controlling for size, n_mod, e2_share, and hier_n:

| Variable | Coefficient | OR | p-value |
|------|-----:|---:|-----:|
| holds | -0.031 | 0.969 | 0.62 |
| size | 2.607 | — | <0.001 |
| n_mod | -0.476 | — | <0.001 |
| e2_share | 0.375 | — | <0.001 |
| hier_n | 0.056 | — | 0.31 |

The independent effect of holds disappears entirely (OR=0.969, p=0.62). What actually drives split is size (positive) and n_mod (negative), plus e2_share (positive).

### Conclusion

The "inverse" in g2 does not mean that criterion b's mechanism fails; it is a **misalignment of the object of the antecedent judgment**: when judging whether "the antecedent of hierarchical unbundling holds", the LLM assigned it to small simple clusters, while the clusters that actually split are the large complex ones.

### Direction of the fix

Adjust the antecedent definition and add size/complexity thresholds. "The antecedent of hierarchical unbundling holds" should not be judged only by whether the task cluster is cohesive and codable; it should also require that the cluster be large enough and contain enough hierarchy levels before it can possibly be absorbed by the upper level and then split.

---

## g3 (Criterion c, mimetic template diffusion) — vacuous shell confirmed; holds has no independent effect

### Phenomenon

The inverse lift ratio is 1.00-1.10; after subtracting the base rate, the net correspondence strength is nearly zero. The base hold rate is 82.5%; the antecedent definition is too broad.

### Diagnosis 1: the p buckets have zero ordering power

Spearman correlations are between -0.01 and -0.06; high p buckets do not correspond to high outcome rates. At p=0.9 the outcome rate is instead the lowest.

| h | Outcome rate at p=0.05 | Outcome rate at p=0.3 | Outcome rate at p=0.5 | Outcome rate at p=0.7 | Outcome rate at p=0.9 | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.154 | 0.225 | 0.218 | 0.181 | 0.143 | -0.01 |
| 2 | 0.000 | 0.242 | 0.222 | 0.200 | 0.067 | +0.01 |
| 3 | 0.500 | 0.135 | 0.128 | 0.118 | 0.000 | -0.06 |

The confidence assigned by the LLM is completely unrelated to the actual outcome rate.

### Diagnosis 2: tightening the antecedent does not work

Binning by p>=0.5 vs p<0.5, the lift ratios are only 1.06/1.18/0.84. With three bins at p>=0.7, the outcome rate of the high-p bucket is instead lower than that of the middle-p bucket.

### Diagnosis 3: logistic regression settles it

After controlling for firm characteristics, holds has OR=1.084 and p=0.64 — no independent effect.

| Variable | Coefficient | p-value |
|------|-----:|-----:|
| holds | 0.081 | 0.64 |
| n_jobs | 0.129 | 0.14 |
| mean_size | -0.065 | 0.63 |
| n_mod | 0.607 | <0.001 |
| surf_sim | -0.339 | 0.089 |

What actually drives above_median is n_mod (number of modules, p<0.01) and surf_sim (surface similarity, p<0.1), not holds.

### Conclusion

g3's problem is more fundamental than g2's: not only is the antecedent definition too broad (82.5% hold rate), but **the LLM's judgments themselves carry no information** — objects to which it assigns high confidence do not have higher outcome rates, and objects to which it assigns low confidence do not have lower outcome rates. Under the current antecedent definition and judgment protocol, criterion c cannot hold.

### Direction of the fix

The antecedent definition needs to be redesigned. Change the antecedent of "mimetic template diffusion" from the LLM's subjective judgment to objective indicators — for example, "the number of templates newly adopted by the firm in 2020" or "the firm's depth of template adoption" — rather than asking the LLM to judge "whether this firm is doing template diffusion".
