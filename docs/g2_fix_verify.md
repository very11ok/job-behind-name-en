# g2 fix verification: new antecedent definition (holds ∧ hier_n≥3)

Threshold source: the threshold scan found that when hier_n≥3 the inverse relationship disappears and the lift ratio jumps from 0.55 to 1.68.

## Conditional probability comparison (by time window h)

| definition            |   h |    n |   n1 |   n0 |   P(split|holds=1) |   P(split|holds=0) |    base rate |   lift ratio |
|:--------------------|----:|-----:|-----:|-----:|-------------------:|-------------------:|-------:|------:|
| Original: holds      |   1 | 1502 |  409 | 1093 |             0.2127 |             0.3715 | 0.3282 |  0.57 |
| Original: holds      |   2 | 1502 |  409 | 1093 |             0.1589 |             0.2818 | 0.2483 |  0.56 |
| Original: holds      |   3 | 1502 |  409 | 1093 |             0.1222 |             0.2498 | 0.215  |  0.49 |
| New: holds ∧ hier_n≥3 |   1 | 1502 |   49 | 1453 |             0.7347 |             0.3145 | 0.3282 |  2.34 |
| New: holds ∧ hier_n≥3 |   2 | 1502 |   49 | 1453 |             0.6122 |             0.2361 | 0.2483 |  2.59 |
| New: holds ∧ hier_n≥3 |   3 | 1502 |   49 | 1453 |             0.5102 |             0.2051 | 0.215  |  2.49 |

## Inverse probability comparison

| definition            |   h |   P(holds=1|split=1) |   P(holds=1|split=0) |   inverse lift ratio |
|:--------------------|----:|---------------------:|---------------------:|--------:|
| Original: holds      |   1 |               0.1765 |               0.3191 |    0.55 |
| Original: holds      |   2 |               0.1743 |               0.3047 |    0.57 |
| Original: holds      |   3 |               0.1548 |               0.3045 |    0.51 |
| New: holds ∧ hier_n≥3 |   1 |               0.073  |               0.0129 |    5.67 |
| New: holds ∧ hier_n≥3 |   2 |               0.0804 |               0.0168 |    4.78 |
| New: holds ∧ hier_n≥3 |   3 |               0.0774 |               0.0204 |    3.8  |

## Chi-square test of independence

- Original: holds, h=1: χ²=34.01, n=1502
- Original: holds, h=2: χ²=24.07, n=1502
- Original: holds, h=3: χ²=28.67, n=1502
- New: holds ∧ hier_n≥3, h=1: χ²=37.95, n=1502
- New: holds ∧ hier_n≥3, h=2: χ²=35.94, n=1502
- New: holds ∧ hier_n≥3, h=3: χ²=26.14, n=1502
