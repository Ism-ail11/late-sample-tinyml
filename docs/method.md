# Exact correction with bounded memory

## Model and stream

The causal network has three width-three convolutions with channel widths 12, 12, and 16 and dilations 1, 4, and 16. A width-one head emits signed INT8 class scores. The maximum temporal span is 43 samples. Every layer has a zero prefix at reset. The same integer kernel runs for appends, replay, and all correction variants.

The normalized acceleration is clipped to [−6, 6] and quantized symmetrically. Each layer has signed INT8 weights, INT32 bias and accumulation, and an INT64 product for requantization. The exported multiplier has 20 fractional bits. Requantization rounds ties away from zero, then clamps to [0, 127] after ReLU or [−127, 127] for scores. The exporter bounds each accumulator for the supported input range. The independent NumPy dense oracle uses the exported params and reconstructs the entire causal graph layer by layer.

A call to `tiny_append` advances one logical timestamp. When a reading is missing, the caller can append the last observed vector or a zero vector, log that the published score is provisional, and later call `tiny_repair` with the true vector and its age in samples. A burst passes several unique ages in one transaction. The caller serializes appends and corrections, associates scores with timestamps, and can read any retained age after the transaction. An age beyond the retained history causes atomic rejection. A repeated age in the same batch returns the duplicate status. Updating a value with an identical INT8 vector is a no-op.

## Dependency propagation

Suppose corrected inputs have changed at times \(S\). The affected times of the next layer are the union of \(S\), \(S+d\), and \(S+2d\), intersected with the already materialized stream prefix. A one-tap head has the same candidate times as its input. The implementation stores this set in a 64-bit mask referenced to the oldest changed time. The default age horizon is 48, so at most 49 times can be repaired in one transaction. The API and tests also support compiled horizons through 63.

For each candidate time, the layer reads the corrected earlier layer and computes the INT8 vector with the ordinary kernel. When equality pruning is enabled, only vectors whose bytes changed enter the next layer's dirty mask. The adaptive variant traverses a dense range when the mask contains at most one hole and traverses set bits otherwise. Batch updates share dirty sets so overlapping dependencies are evaluated once per stage.

For correctness, start with a cached state equal to a fresh evaluation before the update. A corrected value can affect only operators with a path from that input. At each layer, a candidate is evaluated using the exact recomputed parent vectors. Values outside the candidate set have identical parents and the same weights. Equality pruning removes vectors whose exact bytes match the former state, so all their children receive the original bytes. By induction, cached scores match a fresh pass after every accepted transaction. Extra evaluations from the dense traversal preserve the same argument.

Ring capacities differ across layers. A layer with a consumer dilation \(d\) retains \(H+1+2d\) frames, where \(H\) is the maximum repairable age. The current head and logits retain \(H+1\) frames because their consumer has no temporal taps. This keeps the older sources needed for an update exactly at the history boundary and keeps forward inference correct after repair and wraparound.

## Baselines and ablations

| ID | Repair policy |
| --- | --- |
| `replay` | Re-executes every layer in time order from the oldest edited slot to the newest output within the affected receptive-field range. Reuses valid state preceding that range. |
| `span` | Propagates a dense time interval for each layer. |
| `tap` | Propagates the exact timestamp dependency mask. |
| `tap_equal` | Adds byte-exact intermediate equality to the tap mask. |
| `span_equal` | Applies equality pruning to a dense interval at each layer. |
| `adaptive` | Chooses dense or sparse traversal from mask density and enables equality. The complete method. |
| `serial` | Applies the same adaptive method to each member of a validated batch individually. |
| `adaptive_no_equal` | Uses adaptive traversal with equality pruning disabled. |

The seven paired implementation contrasts are generated in Table 3. Table 3 also includes placeholder sensitivity, Table 4 varies the history size, and Table 5 varies the contiguous burst size. A fixed 64-bit representation bounds the storage used by the correction mask.
