# Cross-subject MCU evaluation

## Evaluation units

- WISDM smartwatch acceleration at 20 Hz and MHEALTH right-arm acceleration at 50 Hz retain all 18 and 12 activity classes, respectively.
- The outer test folds partition the subjects using a fixed shuffle. A validation group is formed within the remaining subjects. Each of the three outer folds uses three independently trained seeds (13, 37, 73). The full suite includes 18 classifiers.
- Normalization and quantization calibration use the training partition. Checkpoints use validation macro F1. Test clips are drawn from outer test subjects before timing.
- The primary corpus has three disjoint 120-sample clips per activity per fold, selected to cover every outer test subject. It includes 162 WISDM and 108 MHEALTH clips. The main timing grid uses 21 controlled corrections per clip. The Renode confirmation and resource sweeps select one clip per participant by a fixed hash.
- Singular age sweeps include ages 0 through 8 and selected ages through 48. Other conditions vary fill value, contiguous bursts, scattered bursts, cache size, scalar versus DSP kernels, CPU profile, and expiration horizon. A distinct native stream suite supplies multiple transactions per stream with held-out data, overlapping asynchronous delivery, and explicit expiration. Its pseudorandom schedules are synthetic.

## Fair comparisons

Replay, layer-wise replay, the previous tap mask, and all new variants run from the same firmware source and use the same exported weights and DSP dot product within a comparison. The main timed runs use Unicorn's Cortex-M4 CPU model. A matched Renode Cortex-M4 subset confirms the instruction-cost result, score oracle, and SRAM watermark. An independent NumPy dense reference supplies every retained score byte expected before and after each transaction. The separate native stream suite is an additional correctness check.

Instruction counting includes the measured correction call and its common stack-switch wrapper. It covers validation, traversal, exact integer arithmetic and state updates. It starts after provisional inference and ends before oracle verification and result export. Append cost is measured on the final forward step of every case. Unicorn uses a C instruction hook and Renode uses its CPU executed-instruction counter. They report their own instruction counts, so the comparison table reports their difference. Both are CPU emulators. Hardware energy, pipeline timing, and sensor transport are outside the measured scope.

The two largest layers alone retain recent pre-activation vectors. Report method SRAM as the live C state plus the maximum observed stack watermark. The cache-free build omits the accumulator state. Keep the cache-enabled direct recomputation as the matched compute-only comparison, and use the separate cache-free build for memory and steady-state trade-offs.

## Aggregation

Each repair pairs a workload, participant, clip, model, and correction schedule across variants. For each fold and seed, compute the median of per-case percentage savings. The primary summary averages nine such medians for each dataset. Resource amortization divides mean extra append instructions by mean instructions saved per recent repair on matching cases. To distinguish cache maintenance from code generation, a separate pair of builds prevents the convolution kernel from being inlined or cloned. Both the ordinary builds and the controlled builds are reported. The break-even is modeled under the measured short-repair workload mix.

The optional bootstrap interval in the primary CSV resamples participants within each fixed test fold, retaining all their observations and three seeds. It describes participant variation conditional on the fixed folds, trained models, workloads and firmware. It conditions on the three chosen training partitions. The plots expose performance by correction age and the ablations expose small or negative effects as measured.

The new design and subject split both affect clean accuracy. The firmware comparisons use the same model within each paired trial. Clean held-out accuracy and bit-exact repair measure different outcomes.
