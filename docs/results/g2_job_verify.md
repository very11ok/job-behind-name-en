# Criterion 2 job-version verification (hierarchical unbundling: asymmetry of the job's task set → congealing into new jobs with lower asymmetry)

Objects: 2000 jobs (T=2020); asym = standard deviation of E_ord over the job's task set, median split at 0.7071
Outcome: split = in year T+h, ≥2 narrow jobs appear that carry a subset of its tasks and cover ≥60%; down_std = split and the narrow jobs' asymmetry is lower than the original job's

## Directional outcome down_std
  h=1: P(down_std|asym high)=0.2302(n=986), P(down_std|asym low)=0.0868(n=1014), lift ratio=2.65, chi2=77.51, p=1.32e-18
  h=2: P(down_std|asym high)=0.1582(n=986), P(down_std|asym low)=0.0552(n=1014), lift ratio=2.86, chi2=55.95, p=7.43e-14
  h=3: P(down_std|asym high)=0.1339(n=986), P(down_std|asym low)=0.0454(n=1014), lift ratio=2.95, chi2=48.30, p=3.66e-12

## Control: split
  h=1: P(split|asym high)=0.2454(n=986), P(split|asym low)=0.2150(n=1014), lift ratio=1.14, chi2=2.62, p=0.106
  h=2: P(split|asym high)=0.1775(n=986), P(split|asym low)=0.1548(n=1014), lift ratio=1.15, chi2=1.85, p=0.173
  h=3: P(split|asym high)=0.1410(n=986), P(split|asym low)=0.1243(n=1014), lift ratio=1.13, chi2=1.21, p=0.27

## Narrow jobs with lower mean E down_e
  h=1: P(down_e|asym high)=0.0720(n=986), P(down_e|asym low)=0.1568(n=1014), lift ratio=0.46, chi2=35.32, p=2.8e-09
  h=2: P(down_e|asym high)=0.0609(n=986), P(down_e|asym low)=0.1243(n=1014), lift ratio=0.49, chi2=23.83, p=1.05e-06
  h=3: P(down_e|asym high)=0.0416(n=986), P(down_e|asym low)=0.0986(n=1014), lift ratio=0.42, chi2=24.82, p=6.3e-07

## Robustness: size>=4 subsample (the task set must be large enough to have room to split)
  down_std h=1: 0.2792(n=813) vs 0.1270(n=693), lift ratio=2.20, p=4.5e-13
  down_std h=2: 0.1919(n=813) vs 0.0808(n=693), lift ratio=2.37, p=6.52e-10
  down_std h=3: 0.1624(n=813) vs 0.0664(n=693), lift ratio=2.45, p=8.89e-09
  split h=1: 0.2977(n=813) vs 0.3146(n=693), lift ratio=0.95, p=0.478
  split h=2: 0.2153(n=813) vs 0.2266(n=693), lift ratio=0.95, p=0.598
  split h=3: 0.1710(n=813) vs 0.1818(n=693), lift ratio=0.94, p=0.582
