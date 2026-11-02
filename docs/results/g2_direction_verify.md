# Criterion 2 directional fix verification (asymmetry antecedent → downward unbundling outcome)

Seed clusters: 1502; asym antecedent = e_std > median 0.6960

## Antecedent = down_std (the split-off narrow jobs have lower asymmetry)
  h=1: P(down_std|asym high)=0.2736(n=742), P(down_std|asym low)=0.1750(n=760), lift ratio=1.56, p=4.56e-06
  h=2: P(down_std|asym high)=0.2022(n=742), P(down_std|asym low)=0.1447(n=760), lift ratio=1.40, p=0.00327
  h=3: P(down_std|asym high)=0.1685(n=742), P(down_std|asym low)=0.1303(n=760), lift ratio=1.29, p=0.0377

## Antecedent = down_e (the split-off narrow jobs have lower mean E, routine tasks sinking down to lower-skill work)
  h=1: P(down_e|asym high)=0.0687(n=742), P(down_e|asym low)=0.2737(n=760), lift ratio=0.25, p=7.56e-26
  h=2: P(down_e|asym high)=0.0418(n=742), P(down_e|asym low)=0.2250(n=760), lift ratio=0.19, p=2.34e-25
  h=3: P(down_e|asym high)=0.0377(n=742), P(down_e|asym low)=0.1961(n=760), lift ratio=0.19, p=1.83e-21

## Antecedent = split (original criterion, control)
  h=1: P(split|asym high)=0.2857(n=742), P(split|asym low)=0.3697(n=760), lift ratio=0.77, p=0.000526
  h=2: P(split|asym high)=0.2102(n=742), P(split|asym low)=0.2855(n=760), lift ratio=0.74, p=0.000735
  h=3: P(split|asym high)=0.1725(n=742), P(split|asym low)=0.2566(n=760), lift ratio=0.67, p=7.34e-05

## Cross-tabulation: LLM holds × asym_high → split
  h=1 holds=1 asym_high=1: n=159, split rate=0.1698, down_std rate=0.1572
  h=1 holds=1 asym_high=0: n=250, split rate=0.2400, down_std rate=0.1040
  h=1 holds=0 asym_high=1: n=583, split rate=0.3173, down_std rate=0.3053
  h=1 holds=0 asym_high=0: n=510, split rate=0.4333, down_std rate=0.2098

  h=2 holds=1 asym_high=1: n=159, split rate=0.1069, down_std rate=0.1069
  h=2 holds=1 asym_high=0: n=250, split rate=0.1920, down_std rate=0.0960
  h=2 holds=0 asym_high=1: n=583, split rate=0.2384, down_std rate=0.2281
  h=2 holds=0 asym_high=0: n=510, split rate=0.3314, down_std rate=0.1686

  h=3 holds=1 asym_high=1: n=159, split rate=0.0755, down_std rate=0.0755
  h=3 holds=1 asym_high=0: n=250, split rate=0.1520, down_std rate=0.0720
  h=3 holds=0 asym_high=1: n=583, split rate=0.1990, down_std rate=0.1938
  h=3 holds=0 asym_high=0: n=510, split rate=0.3078, down_std rate=0.1588

## Chi-square (asym_high × down_std)
  h=1: chi2=21.01, p=4.56e-06  (2x2: 203,539,133,627)
  h=2: chi2=8.65, p=0.00327  (2x2: 150,592,110,650)
  h=3: chi2=4.32, p=0.0377  (2x2: 125,617,99,661)

## Combined antecedent: holds=1 ∧ hier_n>=3 ∧ asym_high → down_std
  h=1: P=0.7333(n=15) vs 0.2186(n=1487), lift ratio=3.36, p=1.93e-06
  h=2: P=0.4667(n=15) vs 0.1701(n=1487), lift ratio=2.74, p=0.00253
  h=3: P=0.4000(n=15) vs 0.1466(n=1487), lift ratio=2.73, p=0.00612
