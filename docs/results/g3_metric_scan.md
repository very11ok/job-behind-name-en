# g3 objective metric scan: from LLM judgment to structural metrics

Total sample: 783; n under each h = {1: 292, 2: 232, 3: 259}

## LLM judgment baseline

| h | P(adopt|holds=1) | P(adopt|holds=0) | lift ratio | p value |
|---|-------------------|-------------------|--------|------|
| 1 | 0.1992 | 0.1569 | 1.27 | 4.86e-01 |
| 2 | 0.2053 | 0.1190 | 1.72 | 1.97e-01 |
| 3 | 0.1193 | 0.1220 | 0.98 | 9.61e-01 |

## Conditional probabilities of objective metrics (median split)

| metric              |   h |   n |   n_high |   n_low |   P(adopt|high) |   P(adopt|low) |   lift ratio |       p value |
|:----------------|----:|----:|---------:|--------:|-------------:|-------------:|------:|---------:|
| surface similarity  |   1 | 292 |      147 |     145 |       0.1701 |       0.2138 |  0.8  | 0.343    |
| surface similarity  |   2 | 232 |      114 |     118 |       0.1842 |       0.1949 |  0.95 | 0.835    |
| surface similarity  |   3 | 259 |      131 |     128 |       0.0916 |       0.1484 |  0.62 | 0.159    |
| structural similarity |   1 | 292 |      149 |     143 |       0.2349 |       0.1469 |  1.6  | 0.0561   |
| structural similarity |   2 | 232 |      112 |     120 |       0.1964 |       0.1833 |  1.07 | 0.799    |
| structural similarity |   3 | 259 |      131 |     128 |       0.1298 |       0.1094 |  1.19 | 0.613    |
| cross-firm similarity |   1 | 292 |      148 |     144 |       0.1892 |       0.1944 |  0.97 | 0.909    |
| cross-firm similarity |   2 | 232 |      115 |     117 |       0.1652 |       0.2137 |  0.77 | 0.347    |
| cross-firm similarity |   3 | 259 |      130 |     129 |       0.1    |       0.1395 |  0.72 | 0.327    |
| number of jobs      |   1 | 292 |      144 |     148 |       0.2778 |       0.1081 |  2.57 | 0.000232 |
| number of jobs      |   2 | 232 |      113 |     119 |       0.2743 |       0.1092 |  2.51 | 0.00135  |
| number of jobs      |   3 | 259 |      136 |     123 |       0.1544 |       0.0813 |  1.9  | 0.0703   |
| number of modules   |   1 | 292 |      154 |     138 |       0.2532 |       0.1232 |  2.06 | 0.00483  |
| number of modules   |   2 | 232 |      120 |     112 |       0.2667 |       0.1071 |  2.49 | 0.00195  |
| number of modules   |   3 | 259 |      145 |     114 |       0.1655 |       0.0614 |  2.7  | 0.0104   |
| codable task share  |   1 | 292 |      143 |     149 |       0.2168 |       0.1678 |  1.29 | 0.288    |
| codable task share  |   2 | 232 |      120 |     112 |       0.1917 |       0.1875 |  1.02 | 0.936    |
| codable task share  |   3 | 259 |      129 |     130 |       0.1395 |       0.1    |  1.4  | 0.327    |
| mean job size       |   1 | 292 |      147 |     145 |       0.2313 |       0.1517 |  1.52 | 0.0842   |
| mean job size       |   2 | 232 |      121 |     111 |       0.2066 |       0.1712 |  1.21 | 0.492    |
| mean job size       |   3 | 259 |      137 |     122 |       0.146  |       0.0902 |  1.62 | 0.167    |
| residual (surf-struct) |   1 | 292 |      146 |     146 |       0.1575 |       0.226  |  0.7  | 0.137    |
| residual (surf-struct) |   2 | 232 |      114 |     118 |       0.1754 |       0.2034 |  0.86 | 0.587    |
| residual (surf-struct) |   3 | 259 |      132 |     127 |       0.0909 |       0.1496 |  0.61 | 0.146    |
| residual (surf-xfirm)  |   1 | 292 |      145 |     147 |       0.1793 |       0.2041 |  0.88 | 0.591    |
| residual (surf-xfirm)  |   2 | 232 |      117 |     115 |       0.1966 |       0.1826 |  1.08 | 0.786    |
| residual (surf-xfirm)  |   3 | 259 |      130 |     129 |       0.0923 |       0.1473 |  0.63 | 0.173    |

## Continuous Spearman correlations

| metric              |   h |   Spearman correlation |   n |
|:----------------|----:|-------------:|----:|
| surface similarity  |   1 |      -0.0423 | 292 |
| surface similarity  |   2 |       0.0106 | 232 |
| surface similarity  |   3 |      -0.1124 | 259 |
| structural similarity |   1 |       0.0747 | 292 |
| structural similarity |   2 |      -0.0249 | 232 |
| structural similarity |   3 |      -0.043  | 259 |
| cross-firm similarity |   1 |       0.0484 | 292 |
| cross-firm similarity |   2 |      -0.0806 | 232 |
| cross-firm similarity |   3 |      -0.0364 | 259 |
| number of jobs      |   1 |       0.2418 | 292 |
| number of jobs      |   2 |       0.1973 | 232 |
| number of jobs      |   3 |       0.1631 | 259 |
| number of modules   |   1 |       0.2089 | 292 |
| number of modules   |   2 |       0.213  | 232 |
| number of modules   |   3 |       0.1643 | 259 |
| codable task share  |   1 |       0.0525 | 292 |
| codable task share  |   2 |      -0.0937 | 232 |
| codable task share  |   3 |       0.0007 | 259 |
| mean job size       |   1 |       0.0984 | 292 |
| mean job size       |   2 |       0.0635 | 232 |
| mean job size       |   3 |       0.0729 | 259 |
| residual (surf-struct) |   1 |      -0.0636 | 292 |
| residual (surf-struct) |   2 |      -0.007  | 232 |
| residual (surf-struct) |   3 |      -0.096  | 259 |
| residual (surf-xfirm)  |   1 |      -0.0439 | 292 |
| residual (surf-xfirm)  |   2 |       0.0284 | 232 |
| residual (surf-xfirm)  |   3 |      -0.0935 | 259 |
