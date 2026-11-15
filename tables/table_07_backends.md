Identical ARM firmware in Unicorn and Renode. All score checks and SRAM watermarks agree. Instruction-count differences are relative to Renode. Both experiments retain their original counts.

| Dataset | Method | Matched | Median difference (%) | Max difference (%) |
| --- | ---: | ---: | ---: | ---: |
| WISDM | replay | 255 | 0.066 | 1.980 |
| WISDM | tap_equal | 255 | 0.066 | 1.980 |
| WISDM | recent_replay | 255 | 0.069 | 2.899 |
| WISDM | accumulator | 255 | 0.590 | 2.899 |
| MHEALTH | replay | 50 | 0.063 | 0.990 |
| MHEALTH | tap_equal | 50 | 0.062 | 0.990 |
| MHEALTH | recent_replay | 50 | 0.066 | 1.449 |
| MHEALTH | accumulator | 50 | 0.731 | 1.449 |
