Paired implementation ablations on the listed workload subsets. The CSV includes all fold and seed rows.

| Change | WISDM saved (%) | WISDM wins (%) | MHEALTH saved (%) | MHEALTH wins (%) |
| --- | ---: | ---: | ---: | ---: |
| Age-indexed masks | 0.06 | 91.4 | 0.08 | 96.8 |
| Recent single-update entry | 0.51 | 100.0 | 0.54 | 100.0 |
| Accumulator reuse | 40.08 | 91.0 | 42.18 | 91.1 |
| Recent accumulator dispatch | 0.78 | 100.0 | 0.84 | 100.0 |
| Full replay for three changed taps | 0.00 | 0.0 | 0.00 | 0.0 |
| Exact activation equality | 0.00 | 29.2 | 1.05 | 53.2 |
| Batch application | 60.65 | 95.1 | 51.60 | 100.0 |
| Cached fallback on older samples | -0.00 | 0.0 | -0.00 | 0.0 |
