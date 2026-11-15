# Experimental protocol and interpretation

## Training and trace selection

- **WISDM** uses the smartwatch accelerometer at 20 Hz and retains all 18 activities. Timestamp resets, malformed rows, label changes, and gaps over 250 ms split streams.
- **MHEALTH** uses right-lower-arm acceleration at 50 Hz and retains all 12 labeled activities. Label zero and transitions split the series.
- Each dataset uses one fixed subject-disjoint train/validation/test split shared by training seeds 13, 37, and 73. Per-channel normalization and activation calibration use training data. The checkpoint is selected by validation macro-F1. For clean evaluation, every 96-sample window supplies predictions at indices 42, 50, 58, 66, 74, 82, and 90. Windows from a person stay in one split.
- The native study takes four non-overlapping 120-sample clips per activity, balanced by activity and selected round-robin over available test subjects. Renode uses the first of the same four clips for each activity. The clip list, split seed, replay seed, model weights, and archive hashes are included.
- The benchmark replays the first 108 samples. It holds back selected positions and delivers them together at tick 107. Native tests vary delay, contiguous burst size (2, 4, 8), scattered positions, placeholder, and history limit. Renode's main grid contains five single-sample ages with last-observation fill (1, 8, 16, 32, 48), three ages with zero fill (8, 24, 48), and contiguous bursts of 2, 4, and 8 at ages 8, 24, and 48. At horizons 8 and 24, Renode checks a smaller common subset for expiry and stack use. Earlier frames are available for causal context during every trial.

## Measurements

The dense NumPy reference generates expected provisional and corrected scores. Native and ARM runs check every class score of every retained timestamp before and after each transaction. Expired transactions are checked against the provisional reference, including a status check. A checksum summarizes the latest scores across Renode modes. Unit tests separately exercise multiple correction transactions in one long stream, boundary ages through 63, future appends, extreme INT8 values, and malformed transactions.

Bare-metal code is compiled for Cortex-M4 with Arm GCC 14.2.1 at `-O3` and emulated in Renode 1.17.0. Hooks record the instruction counter at the beginning and end markers. The tightly shared measurement interval includes validation and the common call/stack-switch wrapper. It excludes input transport, initial forecasting, oracle comparisons, output transport, and the watermark scan. The included Thumb program independently confirms that the hook counter reads 11 executed instructions across its known interval. The reported measurements describe instruction counts on the emulated CPU. Physical timing and energy require hardware measurements.

Live method SRAM equals the compiler's `sizeof(TinyState)` plus the maximum observed repair stack watermark. The stack is watermarked in a separate 1 KiB benchmark buffer. Fixture data, caller-owned input, the replay harness, and unused watermarked capacity are outside this metric. SRAM numbers describe observed use for these trials and configurations.

Native kernel MACs are the observed number of calls at each layer multiplied by that layer's weight count. These counts cover convolution arithmetic. Indexing, mask traversal, branching, and comparisons are covered by the Renode instruction metric. The Renode estimates include those instruction costs. Report Renode values for the main performance result.

For a paired trial the savings are `100 × (1 − method_work / reference_work)`. For each seed and dataset, the analysis summarizes trial-level ratios first, then aggregates the three seed medians with a sample standard deviation. A win uses a strictly positive savings value. A zero MAC denominator is excluded from a relative MAC comparison. The three seeds vary model initialization and minibatch order on the same held-out people, so seed variability describes training runs on this split. The memory-coverage percentages refer to the same artificial delay distribution across all three budgets.

## Reading the results

The primary claim is exact score correction on the evaluated INT8 networks, with lower median instruction cost than temporal replay on both datasets. Temporal tap dependencies and batching account for most of the savings. Equality pruning is data-dependent. The adaptive-plus-equality version can be slightly slower than replay for short corrections, and the equality check slightly increases the median cost compared with its disabled ablation on these inputs. Table 1 reports clean held-out human activity classification performance, including accuracy and macro-F1.

The study uses bare-metal Cortex-M4 emulation on public activity traces with controlled, injected delivery delays. It supports claims for the evaluated model, workload, compiler, and CPU profile. Natural network jitter, end-to-end power and timing, alternate kernels, other models, and new subject splits are outside the measured scope.
