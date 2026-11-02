# Inverse probability table of the three criteria "outcome → antecedent" (T=2020)

The forward and inverse tables are Bayesian inverses of each other: the forward table looks at P(outcome|holds), the inverse at P(holds|outcome).
Inverse lift ratio = P(holds=1|outcome=1) / P(holds=1|outcome=0).

Interpretation:
  >1 and far above the base rate → the antecedent holds more often when the outcome occurs; the criterion has explanatory power
  ≈1 and close to the base rate → the antecedent and the outcome are essentially independent; the criterion has no discriminating power
  <1 → the antecedent is less likely to hold when the outcome occurs; an inverse relationship

When the base rate (P(holds=1)) is high, be wary: even if the criterion has no explanatory power, P(holds=1|outcome=1) will be pulled up.
What is truly informative is the "inverse lift ratio" and the "net correspondence after deducting the base rate".

## Inverse probability table

| rule   |   h |    n |   P(holds=1|outcome=1) |   P(holds=1|outcome=0) |   base hold rate |   inverse lift ratio |   n_outcome1 |   n_outcome0 |
|:-------|----:|-----:|-----------------------:|-----------------------:|--------:|--------:|-------------:|-------------:|
| g1     |   1 | 2000 |                 0.6491 |                 0.4313 |  0.4375 |    1.51 |           57 |         1943 |
| g1     |   2 | 2000 |                 0.7    |                 0.4362 |  0.4375 |    1.6  |           10 |         1990 |
| g1     |   3 | 2000 |                 0.6667 |                 0.4372 |  0.4375 |    1.53 |            3 |         1997 |
| g2     |   1 | 1502 |                 0.1765 |                 0.3191 |  0.2723 |    0.55 |          493 |         1009 |
| g2     |   2 | 1502 |                 0.1743 |                 0.3047 |  0.2723 |    0.57 |          373 |         1129 |
| g2     |   3 | 1502 |                 0.1548 |                 0.3045 |  0.2723 |    0.51 |          323 |         1179 |
| g3     |   1 |  292 |                 0.8571 |                 0.8178 |  0.8253 |    1.05 |           56 |          236 |
| g3     |   2 |  232 |                 0.8864 |                 0.8032 |  0.819  |    1.1  |           44 |          188 |
| g3     |   3 |  259 |                 0.8387 |                 0.8421 |  0.8417 |    1    |           31 |          228 |

## Forward table comparison (lift ratio)

| rule | h | forward lift ratio | inverse lift ratio | base hold rate |
|:-----|---:|----------:|----------:|----------:|
| g1 | 1 | 2.38 | 1.51 | 0.4375 |
| g1 | 2 | 3.0 | 1.6 | 0.4375 |
| g1 | 3 | 2.57 | 1.53 | 0.4375 |
| g2 | 1 | 0.57 | 0.55 | 0.2723 |
| g2 | 2 | 0.56 | 0.57 | 0.2723 |
| g2 | 3 | 0.49 | 0.51 | 0.2723 |
| g3 | 1 | 1.27 | 1.05 | 0.8253 |
| g3 | 2 | 1.72 | 1.1 | 0.819 |
| g3 | 3 | 0.98 | 1.0 | 0.8417 |
