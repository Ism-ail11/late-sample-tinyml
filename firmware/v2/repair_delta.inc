/* Included by tinyml.c. Only the two expensive ReLU layers cache accumulators. */

typedef struct {
    _Alignas(4) int8_t vector[TINY_DELTA_FRAMES][TINY_MAX_WIDTH];
    int8_t slot[TINY_ACC_FRAMES + 2 * (TINY_D1 > TINY_D2 ? TINY_D1 : TINY_D2)];
    uint8_t size;
} TinyDeltaLog;

static inline void clear_log(TinyDeltaLog *log)
{
    for (unsigned i = 0; i < sizeof(log->slot); ++i) {
        log->slot[i] = -1;
    }
    log->size = 0;
}

static inline int8_t *log_output(TinyDeltaLog *log, unsigned age)
{
    if (!log || age >= sizeof(log->slot) || log->size == TINY_DELTA_FRAMES) {
        return NULL;
    }
    unsigned id = log->size++;
    log->slot[age] = (int8_t)id;
    return log->vector[id];
}

static inline bool delta_old(int32_t *acc, int8_t *dest, const TinyLayerParams *params,
                             unsigned width, unsigned outputs, const unsigned *taps,
                             const int8_t *const *vectors, unsigned count, int8_t *out)
{
    unsigned differs = 0;
    for (unsigned oc = 0; oc < outputs; ++oc) {
        const int8_t *base = params->weight + oc * 3u * width;
        int32_t sum = acc[oc];
        for (unsigned j = 0; j < count; ++j) {
            sum = tiny_dot(sum, vectors[j], base + taps[j] * width, width);
        }
        acc[oc] = sum;
        const int8_t next = requantize(sum, params->multiplier, true);
        int8_t change = (int8_t)((int)next - dest[oc]);
        if (out) {
            out[oc] = change;
        }
        differs |= (uint8_t)change;
        dest[oc] = next;
    }
    return differs != 0;
}

static bool repair_site(TinyState *state, unsigned layer, unsigned age, uint64_t dirty,
                        const TinyDeltaLog *prior, TinyDeltaLog *later, unsigned max_taps)
{
    const TinyLayerParams *params = &tiny_layers[layer];
    int8_t *out = log_output(later, age);
    int32_t *acc = accumulator_frame(state, layer, age);
    if (acc) {
        unsigned changed_taps = 0;
        unsigned taps[3];
        const int8_t *vectors[3];
        for (unsigned tap = 0; tap < 3; ++tap) {
            const unsigned input_age = age + tap * params->dilation;
            if (input_age >= TINY_MAX_REPAIR || !(dirty & (UINT64_C(1) << input_age))) {
                continue;
            }
            const int id = prior->slot[input_age];
            if (id < 0 || changed_taps == max_taps) {
                return evaluate_delta(state, layer, age, true, out);
            }
            taps[changed_taps] = tap;
            vectors[changed_taps] = prior->vector[(unsigned)id];
            ++changed_taps;
        }
        if (changed_taps) {
            return delta_old(acc, frame(state, layer + 1, age), params, channels[layer],
                             channels[layer + 1], taps, vectors, changed_taps, out);
        }
    }
    return evaluate_delta(state, layer, age, layer + 1 < TINY_LAYERS, out);
}

__attribute__((noinline))
static void delta_layers(TinyState *state, uint64_t dirty, TinyMode mode)
{
    TinyDeltaLog buffers[2];
    TinyDeltaLog *prior = &buffers[0], *later = &buffers[1];
    const unsigned max_taps = mode == TINY_DELTA_ALL ? 3 : 2;
    const bool compare = mode != TINY_DELTA_NO_EQUAL;
    for (unsigned layer = 0; layer < TINY_LAYERS && dirty; ++layer) {
        TinyDeltaLog *output = layer < 2 ? later : NULL;
        if (output) {
            clear_log(output);
        }
        const uint64_t candidates = age_footprint(dirty, &tiny_layers[layer]);
        uint64_t pending = candidates;
        uint64_t next = 0;
        while (pending) {
            unsigned age = first_bit(pending);
            pending &= pending - 1u;
            if (repair_site(state, layer, age, dirty, prior, output, max_taps)) {
                next |= UINT64_C(1) << age;
            }
        }
        dirty = compare ? next : candidates;
        TinyDeltaLog *swap = prior;
        prior = later;
        later = swap;
    }
}

__attribute__((noinline))
static void delta_short_single(TinyState *state, unsigned age, TinyMode mode)
{
    TinyDeltaLog buffers[2];
    TinyDeltaLog *prior = &buffers[0], *later = &buffers[1];
    uint32_t dirty = UINT32_C(1) << age;
    const unsigned max_taps = mode == TINY_DELTA_ALL ? 3 : 2;
    const bool compare = mode != TINY_DELTA_NO_EQUAL;
    for (unsigned layer = 0; layer < TINY_LAYERS && dirty; ++layer) {
        TinyDeltaLog *output = layer < 2 ? later : NULL;
        if (output) {
            output->size = 0;
            for (unsigned i = 0; i <= age && i < sizeof(output->slot); ++i) {
                output->slot[i] = -1;
            }
        }
        const TinyLayerParams *params = &tiny_layers[layer];
        uint32_t candidates = dirty;
        if (params->taps > 1 && params->dilation <= age) {
            unsigned d = params->dilation;
            candidates |= (dirty >> d) | (dirty >> (2u * d));
        }
        uint32_t pending = candidates;
        uint32_t next = 0;
        while (pending) {
            unsigned at = (unsigned)__builtin_ctz(pending);
            pending &= pending - 1u;
            if (repair_site(state, layer, at, dirty, prior, output, max_taps)) {
                next |= UINT32_C(1) << at;
            }
        }
        dirty = compare ? next : candidates;
        TinyDeltaLog *swap = prior;
        prior = later;
        later = swap;
    }
}
