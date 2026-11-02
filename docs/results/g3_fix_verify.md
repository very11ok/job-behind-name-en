# Criterion 3 fix verification: objective size antecedent → template adoption outcome

Sample: 783 firm×h; n_jobs median=13.0, n_mod median=14.0

## Antecedent = jobs_high
  h=1: P(high adoption|antecedent=1)=0.2733(n=150), P(high adoption|antecedent=0)=0.1056(n=142), lift ratio=2.59, p=0.000275
        adopt_share mean: 0.0072 vs 0.0095
  h=2: P(high adoption|antecedent=1)=0.2672(n=116), P(high adoption|antecedent=0)=0.1121(n=116), lift ratio=2.38, p=0.00257
        adopt_share mean: 0.0096 vs 0.0121
  h=3: P(high adoption|antecedent=1)=0.1571(n=140), P(high adoption|antecedent=0)=0.0756(n=119), lift ratio=2.08, p=0.044
        adopt_share mean: 0.0055 vs 0.0050

## Antecedent = mod_high
  h=1: P(high adoption|antecedent=1)=0.2969(n=128), P(high adoption|antecedent=0)=0.1098(n=164), lift ratio=2.70, p=5.58e-05
        adopt_share mean: 0.0075 vs 0.0090
  h=2: P(high adoption|antecedent=1)=0.2952(n=105), P(high adoption|antecedent=0)=0.1024(n=127), lift ratio=2.88, p=0.000191
        adopt_share mean: 0.0106 vs 0.0110
  h=3: P(high adoption|antecedent=1)=0.1885(n=122), P(high adoption|antecedent=0)=0.0584(n=137), lift ratio=3.23, p=0.00128
        adopt_share mean: 0.0064 vs 0.0042

## Antecedent = combo
  h=1: P(high adoption|antecedent=1)=0.3140(n=121), P(high adoption|antecedent=0)=0.1053(n=171), lift ratio=2.98, p=8.04e-06
        adopt_share mean: 0.0079 vs 0.0086
  h=2: P(high adoption|antecedent=1)=0.3163(n=98), P(high adoption|antecedent=0)=0.0970(n=134), lift ratio=3.26, p=2.57e-05
        adopt_share mean: 0.0114 vs 0.0104
  h=3: P(high adoption|antecedent=1)=0.1913(n=115), P(high adoption|antecedent=0)=0.0625(n=144), lift ratio=3.06, p=0.00151
        adopt_share mean: 0.0067 vs 0.0041

## Antecedent = holds
  h=1: P(high adoption|antecedent=1)=0.1992(n=241), P(high adoption|antecedent=0)=0.1569(n=51), lift ratio=1.27, p=0.486
        adopt_share mean: 0.0080 vs 0.0096
  h=2: P(high adoption|antecedent=1)=0.2053(n=190), P(high adoption|antecedent=0)=0.1190(n=42), lift ratio=1.72, p=0.197
        adopt_share mean: 0.0103 vs 0.0133
  h=3: P(high adoption|antecedent=1)=0.1193(n=218), P(high adoption|antecedent=0)=0.1220(n=41), lift ratio=0.98, p=0.961
        adopt_share mean: 0.0050 vs 0.0068

## Chi-square (antecedent × above_median)
  jobs_high:  h=1 chi2=13.24 p=0.000275  h=2 chi2=9.09 p=0.00257  h=3 chi2=4.06 p=0.044
  mod_high:  h=1 chi2=16.24 p=5.58e-05  h=2 chi2=13.91 p=0.000191  h=3 chi2=10.37 p=0.00128
  combo:  h=1 chi2=19.93 p=8.04e-06  h=2 chi2=17.71 p=2.57e-05  h=3 chi2=10.07 p=0.00151
  holds:  h=1 chi2=0.49 p=0.486  h=2 chi2=1.66 p=0.197  h=3 chi2=0.00 p=0.961

## Inverse probability P(antecedent=1 | outcome)
  h=1 jobs_high: P(antecedent|high adoption)=0.7321 vs P(antecedent|low adoption)=0.4619, inverse lift ratio=1.59
  h=2 jobs_high: P(antecedent|high adoption)=0.7045 vs P(antecedent|low adoption)=0.4521, inverse lift ratio=1.56
  h=3 jobs_high: P(antecedent|high adoption)=0.7097 vs P(antecedent|low adoption)=0.5175, inverse lift ratio=1.37
  h=1 mod_high: P(antecedent|high adoption)=0.6786 vs P(antecedent|low adoption)=0.3814, inverse lift ratio=1.78
  h=2 mod_high: P(antecedent|high adoption)=0.7045 vs P(antecedent|low adoption)=0.3936, inverse lift ratio=1.79
  h=3 mod_high: P(antecedent|high adoption)=0.7419 vs P(antecedent|low adoption)=0.4342, inverse lift ratio=1.71
  h=1 combo: P(antecedent|high adoption)=0.6786 vs P(antecedent|low adoption)=0.3517, inverse lift ratio=1.93
  h=2 combo: P(antecedent|high adoption)=0.7045 vs P(antecedent|low adoption)=0.3564, inverse lift ratio=1.98
  h=3 combo: P(antecedent|high adoption)=0.7097 vs P(antecedent|low adoption)=0.4079, inverse lift ratio=1.74
  h=1 holds: P(antecedent|high adoption)=0.8571 vs P(antecedent|low adoption)=0.8178, inverse lift ratio=1.05
  h=2 holds: P(antecedent|high adoption)=0.8864 vs P(antecedent|low adoption)=0.8032, inverse lift ratio=1.10
  h=3 holds: P(antecedent|high adoption)=0.8387 vs P(antecedent|low adoption)=0.8421, inverse lift ratio=1.00

## Continuous Spearman (metrics vs adopt_share, per h)
  h=1 n_jobs: rho=0.2418 (n=291)
  h=2 n_jobs: rho=0.1973 (n=232)
  h=3 n_jobs: rho=0.1631 (n=259)
  h=1 n_mod: rho=0.2089 (n=291)
  h=2 n_mod: rho=0.2130 (n=232)
  h=3 n_mod: rho=0.1643 (n=259)
  h=1 e2_share: rho=0.0525 (n=291)
  h=2 e2_share: rho=-0.0937 (n=232)
  h=3 e2_share: rho=0.0007 (n=259)
  h=1 mean_size: rho=0.0984 (n=291)
  h=2 mean_size: rho=0.0635 (n=232)
  h=3 mean_size: rho=0.0729 (n=259)
  h=1 struct_sim: rho=0.0747 (n=291)
  h=2 struct_sim: rho=-0.0249 (n=232)
  h=3 struct_sim: rho=-0.0430 (n=259)
  h=1 surf_sim: rho=-0.0423 (n=291)
  h=2 surf_sim: rho=0.0106 (n=232)
  h=3 surf_sim: rho=-0.1124 (n=259)
