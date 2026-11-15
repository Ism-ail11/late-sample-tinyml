# Exact repair with a short accumulator cache

The application receives a timestamped accelerometer vector at each logical step. It can fill a missing slot with the last available reading, emit a provisional INT8 prediction, and refine retained predictions when a delayed packet arrives. This study treats timestamps as sample indices. The primary experiment uses controlled packet delivery on recorded signals.

## What changed in the firmware

1. A 64-bit dirty mask is indexed by age. If a layer has temporal taps at offsets 0, d, and 2d, its affected output ages are the original mask and the masks shifted right by d and 2d. This represents the time direction directly. An isolated correction younger than eight steps has a shorter 32-bit dispatch path.
2. The firmware retains the eight latest INT32 pre-activation vectors for the two wider ReLU layers. When an eligible output has an available accumulator and a record of each changed input activation, it adds only the products for the changed taps. If a needed delta is outside the bounded scratch log or all three taps change, the main implementation computes the entire output from the current inputs.
3. Each updated accumulator is requantized by the shared INT8 kernel. An activation reaches the next dirty mask when any channel changes. A batch unions overlapping dependencies. The final pointwise classification layer runs an ordinary dot product.
4. On Cortex-M4 and Cortex-M33 builds with the DSP extension, the common kernel uses pairs of signed multiply-accumulates. A scalar configuration supports the same arithmetic and provides a portability check.

## Why the scores are exact

For one output channel, the old accumulator is the bias plus the dot products across all old parents. The new accumulator is the old accumulator plus the dot products of the weights with the input differences. The cached layers receive inputs from clipped ReLU outputs in [0,127]. Each signed difference lies in [−127,127], so the sparse dot product uses signed bytes and the arithmetic remains exact. The exporter checks that the weights and bias admit INT32 accumulation. The unchanged fixed-point multiplier, rounding rule, and clamp produce the same output byte as dense inference. A missing cache entry or scratch record uses the same full kernel.

Each layer reads revised parents before recomputing its output. Earlier times outside the dependency set keep their previously verified scores. Cached accumulators are refreshed on appends and on all full evaluations. The transaction validates all ages and duplicates before editing any inputs. Repeated revisions, overwritten ring slots, and strictly expired transactions are exercised by the tests.

## Scope of the contribution

Prior work already establishes streaming convolution reuse and sparse change propagation. The focus here is an MCU-specific, bounded INT32 cache for **retroactively** updating INT8 predictions, together with a measurement of the trade-off between repair cost, steady-state append overhead, and live SRAM. The original implementation and its archived measurements remain reproducible in the repository. The literature discussion belongs with the embedded correction problem and its nearest baselines.

Related sources are [StreamEase](https://doi.org/10.1109/ICECS61496.2024.10848742), [DeltaCNN](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html), and [Arm CMSIS-NN documentation](https://arm-software.github.io/CMSIS_6/main/NN/index.html).
