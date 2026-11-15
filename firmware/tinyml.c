#include "tinyml.h"
#include "model_params.h"
#ifdef TINY_EXTENDED
#include "simd.h"
#endif

static const uint16_t frames[] = {TINY_RING0, TINY_RING1, TINY_RING2, TINY_RING3, TINY_RING4};
static const uint8_t channels[] = {TINY_INPUTS, TINY_C0, TINY_C1, TINY_C2, TINY_CLASSES};

#ifdef TINY_COUNT_CALLS
uint32_t tiny_layer_calls[TINY_LAYERS];
#endif

static inline size_t slot(const TinyState *state, unsigned layer, unsigned age)
{
    int index = (int)state->head[layer] - (int)age;
    return (size_t)(index < 0 ? index + frames[layer] : index);
}

static inline int8_t *ring(TinyState *state, unsigned layer)
{
    switch (layer) {
    case 0: return &state->input[0][0];
    case 1: return &state->hidden0[0][0];
    case 2: return &state->hidden1[0][0];
    case 3: return &state->hidden2[0][0];
    default: return &state->scores[0][0];
    }
}

static inline int8_t *frame(TinyState *state, unsigned layer, unsigned age)
{
    return ring(state, layer) + slot(state, layer, age) * channels[layer];
}

static inline int8_t requantize(int32_t accumulator, int32_t multiplier, bool relu)
{
    int64_t product = (int64_t)accumulator * multiplier;
    uint64_t magnitude = (uint64_t)(product < 0 ? -product : product);
    int64_t rounded = (int64_t)((magnitude + (1u << (TINY_SHIFT - 1))) >> TINY_SHIFT);
    if (product < 0) {
        rounded = -rounded;
    }
    int low = relu ? 0 : -127;
    if (rounded > 127) {
        return 127;
    }
    if (rounded < low) {
        return (int8_t)low;
    }
    return (int8_t)rounded;
}

#if defined(TINY_EXTENDED) && TINY_ACC_FRAMES > 0
static inline int32_t *accumulator_frame(TinyState *state, unsigned layer, unsigned age)
{
    if (age >= TINY_ACC_FRAMES) {
        return NULL;
    }
    unsigned index = ((unsigned)state->accumulator_head - age) & (TINY_ACC_FRAMES - 1u);
    if (layer == 1) {
        return state->acc1[index];
    }
    if (layer == 2) {
        return state->acc2[index];
    }
    return NULL;
}
#endif

/* The optional difference vector is used for ReLU-layer outputs, whose
 * updated and previous INT8 values both lie in [0, 127]. */
#ifdef TINY_EXTENDED
#ifdef TINY_LOCK_KERNEL
__attribute__((noipa))
#endif
static bool evaluate_delta(TinyState *state, unsigned layer, unsigned age, bool compare, int8_t *delta)
#else
static bool evaluate(TinyState *state, unsigned layer, unsigned age, bool compare)
#endif
{
#ifdef TINY_COUNT_CALLS
    ++tiny_layer_calls[layer];
#endif
    const TinyLayerParams *params = &tiny_layers[layer];
    const unsigned in = channels[layer];
    const unsigned out = channels[layer + 1];
    const int8_t *sources[3];
    for (unsigned tap = 0; tap < params->taps; ++tap) {
        sources[tap] = frame(state, layer, age + tap * params->dilation);
    }
    int8_t *dest = frame(state, layer + 1, age);
    const int8_t *weight = params->weight;
    unsigned different = 0;
#if defined(TINY_EXTENDED) && TINY_ACC_FRAMES > 0
    int32_t *acc = accumulator_frame(state, layer, age);
#endif
    for (unsigned oc = 0; oc < out; ++oc) {
        int32_t sum = params->bias[oc];
        for (unsigned tap = 0; tap < params->taps; ++tap) {
            const int8_t *source = sources[tap];
#ifdef TINY_EXTENDED
            sum = tiny_dot(sum, source, weight, in);
            weight += in;
#else
            for (unsigned ic = 0; ic < in; ++ic) {
                sum += (int32_t)source[ic] * *weight++;
            }
#endif
        }
#if defined(TINY_EXTENDED) && TINY_ACC_FRAMES > 0
        if (acc) {
            acc[oc] = sum;
        }
#endif
        int8_t next = requantize(sum, params->multiplier, params->relu);
#ifdef TINY_EXTENDED
        if (delta) {
            int8_t change = (int8_t)((int)next - dest[oc]);
            delta[oc] = change;
            different |= (uint8_t)change;
        } else
#endif
        if (compare) {
            different |= (uint8_t)(dest[oc] ^ next);
        }
        dest[oc] = next;
    }
    return different != 0;
}

#ifdef TINY_EXTENDED
static bool evaluate(TinyState *state, unsigned layer, unsigned age, bool compare)
{
    return evaluate_delta(state, layer, age, compare, NULL);
}
#endif

void tiny_reset(TinyState *state)
{
    unsigned char *bytes = (unsigned char *)state;
    for (size_t i = 0; i < sizeof(*state); ++i) {
        bytes[i] = 0;
    }
    for (unsigned layer = 0; layer <= TINY_LAYERS; ++layer) {
        state->head[layer] = frames[layer] - 1;
    }
}

void tiny_append(TinyState *state, const int8_t sample[TINY_INPUTS])
{
    for (unsigned layer = 0; layer <= TINY_LAYERS; ++layer) {
        unsigned next = state->head[layer] + 1u;
        state->head[layer] = (uint16_t)(next == frames[layer] ? 0 : next);
    }
#if defined(TINY_EXTENDED) && TINY_ACC_FRAMES > 0
    state->accumulator_head = (uint8_t)((state->accumulator_head + 1u) & (TINY_ACC_FRAMES - 1u));
#endif
    if (state->filled < TINY_MAX_REPAIR) {
        ++state->filled;
    }
    int8_t *dest = frame(state, 0, 0);
    for (unsigned i = 0; i < TINY_INPUTS; ++i) {
        dest[i] = sample[i];
    }
    for (unsigned layer = 0; layer < TINY_LAYERS; ++layer) {
        evaluate(state, layer, 0, false);
    }
}

const int8_t *tiny_scores(const TinyState *state, uint8_t age)
{
    if (age >= state->filled) {
        return NULL;
    }
    return state->scores[slot(state, TINY_LAYERS, age)];
}

static inline uint64_t through(unsigned last)
{
    return UINT64_MAX >> (63u - last);
}

static inline unsigned first_bit(uint64_t value)
{
    uint32_t low = (uint32_t)value;
    return low ? (unsigned)__builtin_ctz(low) : 32u + (unsigned)__builtin_ctz((uint32_t)(value >> 32));
}

static inline unsigned last_bit(uint64_t value)
{
    uint32_t high = (uint32_t)(value >> 32);
    return high ? 63u - (unsigned)__builtin_clz(high) : 31u - (unsigned)__builtin_clz((uint32_t)value);
}

static inline uint64_t layer_footprint(uint64_t dirty, const TinyLayerParams *params)
{
    if (params->taps == 1) {
        return dirty;
    }
    unsigned dilation = params->dilation;
    return dirty | (dirty << dilation) | (dirty << (2u * dilation));
}

static void replay(TinyState *state, unsigned oldest, unsigned youngest)
{
    /* Reuse exact older state. The latest affected output is bounded by the receptive field. */
    unsigned last_age = youngest > TINY_RF - 1 ? youngest - (TINY_RF - 1) : 0;
    for (unsigned age = oldest + 1; age-- > last_age;) {
        for (unsigned layer = 0; layer < TINY_LAYERS; ++layer) {
            evaluate(state, layer, age, false);
        }
    }
}

static void correct_layers(TinyState *state, uint64_t dirty, unsigned oldest, TinyMode mode)
{
    const uint64_t valid = through(oldest);
    for (unsigned layer = 0; layer < TINY_LAYERS && dirty; ++layer) {
        uint64_t candidates = layer_footprint(dirty, &tiny_layers[layer]) & valid;
        const bool compare = layer + 1 < TINY_LAYERS &&
                             (mode == TINY_TIME_EQUAL || mode == TINY_SPAN_EQUAL || mode == TINY_ADAPTIVE);
        uint64_t changed = 0;

        if (mode == TINY_SPAN || mode == TINY_SPAN_EQUAL ||
            mode == TINY_ADAPTIVE || mode == TINY_ADAPTIVE_NO_EQUAL) {
            const unsigned first = first_bit(candidates);
            const unsigned last = last_bit(candidates);
            const uint64_t span = through(last) & (UINT64_MAX << first);
            const uint64_t holes = span ^ candidates;
            /* Dense bounds avoid bit extraction when there is at most one hole. */
            const bool use_density = mode == TINY_ADAPTIVE || mode == TINY_ADAPTIVE_NO_EQUAL;
            const bool dense = !use_density || (holes & (holes - 1u)) == 0;
            if (dense) {
                for (unsigned bit = first; bit <= last; ++bit) {
                    if (evaluate(state, layer, oldest - bit, compare)) {
                        changed |= UINT64_C(1) << bit;
                    }
                }
                dirty = compare ? changed : span;
                continue;
            }
        }

        const uint64_t original = candidates;
        while (candidates) {
            unsigned bit = first_bit(candidates);
            candidates &= candidates - 1u;
            if (evaluate(state, layer, oldest - bit, compare)) {
                changed |= UINT64_C(1) << bit;
            }
        }
        dirty = compare ? changed : original;
    }
}

#ifdef TINY_EXTENDED
/* A bit denotes an age. Causality moves a dirty input to younger ages. */
static inline uint64_t age_footprint(uint64_t mask, const TinyLayerParams *params)
{
    if (params->taps == 1) {
        return mask;
    }
    unsigned d = params->dilation;
    return mask | (mask >> d) | (mask >> (2u * d));
}

static void repair_ages(TinyState *state, uint64_t dirty, bool compare)
{
    for (unsigned layer = 0; layer < TINY_LAYERS && dirty; ++layer) {
        const uint64_t candidates = age_footprint(dirty, &tiny_layers[layer]);
        uint64_t pending = candidates;
        uint64_t changed = 0;
        while (pending) {
            unsigned age = first_bit(pending);
            pending &= pending - 1u;
            if (evaluate(state, layer, age, compare && layer + 1 < TINY_LAYERS)) {
                changed |= UINT64_C(1) << age;
            }
        }
        dirty = compare ? changed : candidates;
    }
}

static void repair_recent(TinyState *state, unsigned age, bool compare)
{
    uint32_t dirty = UINT32_C(1) << age;
    for (unsigned layer = 0; layer < TINY_LAYERS && dirty; ++layer) {
        const TinyLayerParams *params = &tiny_layers[layer];
        uint32_t candidates = dirty;
        if (params->taps > 1 && params->dilation <= age) {
            unsigned d = params->dilation;
            candidates |= (dirty >> d) | (dirty >> (2u * d));
        }
        uint32_t pending = candidates;
        uint32_t changed = 0;
        while (pending) {
            unsigned at = (unsigned)__builtin_ctz(pending);
            pending &= pending - 1u;
            if (evaluate(state, layer, at, compare && layer + 1 < TINY_LAYERS)) {
                changed |= UINT32_C(1) << at;
            }
        }
        dirty = compare ? changed : candidates;
    }
}

static bool replace_input(TinyState *state, const TinyUpdate *update)
{
    int8_t *dest = frame(state, 0, update->age);
    unsigned differs = 0;
    for (unsigned i = 0; i < TINY_INPUTS; ++i) {
        differs |= (uint8_t)(dest[i] ^ update->sample[i]);
        dest[i] = update->sample[i];
    }
    return differs != 0;
}

#if TINY_ACC_FRAMES > 0
#include "v2/repair_delta.inc"
#endif
#endif

static void apply_updates(TinyState *state, const TinyUpdate *updates, size_t count, TinyMode mode)
{
    uint64_t ages = 0;
    for (size_t index = 0; index < count; ++index) {
        const TinyUpdate *update = &updates[index];
#ifdef TINY_EXTENDED
        if (replace_input(state, update)) {
            ages |= UINT64_C(1) << update->age;
        }
#else
        int8_t *dest = frame(state, 0, update->age);
        unsigned changed = 0;
        for (unsigned axis = 0; axis < TINY_INPUTS; ++axis) {
            changed |= (uint8_t)(dest[axis] ^ update->sample[axis]);
            dest[axis] = update->sample[axis];
        }
        if (changed) {
            ages |= UINT64_C(1) << update->age;
        }
#endif
    }
    if (!ages) {
        return;
    }

    const unsigned oldest = last_bit(ages);
#ifdef TINY_EXTENDED
    if (mode == TINY_AGE_MASK || mode == TINY_AGE_EQUAL || mode == TINY_RECENT_REPLAY) {
        repair_ages(state, ages, mode != TINY_AGE_MASK);
        return;
    }
    if (mode == TINY_DIRECT_REPLAY) {
        // Recompute the touched interval layer by layer. Each layer needs
        // only the observed downstream range of its own receptive field.
        unsigned extent = 0;
        unsigned youngest = first_bit(ages);
        for (unsigned layer = 0; layer < TINY_LAYERS; ++layer) {
            const TinyLayerParams *params = &tiny_layers[layer];
            extent += (params->taps - 1u) * params->dilation;
            unsigned limit = youngest > extent ? youngest - extent : 0;
            for (unsigned age = oldest + 1u; age-- > limit;) {
                evaluate(state, layer, age, false);
            }
        }
        return;
    }
    if (mode >= TINY_DELTA && mode <= TINY_DELTA_NO_EQUAL) {
#if TINY_ACC_FRAMES > 0
        if (mode != TINY_DELTA_DISABLED && (ages & ((UINT64_C(1) << TINY_ACC_FRAMES) - 1))) {
            delta_layers(state, ages, mode);
            return;
        }
#endif
        repair_ages(state, ages, mode != TINY_DELTA_NO_EQUAL);
        return;
    }
#endif
    if (mode == TINY_REPLAY) {
        replay(state, oldest, first_bit(ages));
        return;
    }
    /* Input discovery uses ages. Propagation uses increasing time from the oldest change. */
    uint64_t dirty = 0;
    while (ages) {
        unsigned age = first_bit(ages);
        ages &= ages - 1u;
        dirty |= UINT64_C(1) << (oldest - age);
    }
    correct_layers(state, dirty, oldest, mode);
}

TinyStatus tiny_repair(TinyState *state, const TinyUpdate *updates, size_t count, TinyMode mode)
{
    if ((unsigned)mode >= TINY_MODE_COUNT || count > TINY_MAX_REPAIR || (count && !updates)) {
        return TINY_INVALID;
    }
#ifdef TINY_EXTENDED
    if (count == 1 && mode >= TINY_DELTA && mode != TINY_DELTA_GENERIC &&
        updates[0].age < TINY_DELTA_SHORT_FRAMES) {
        if (updates[0].age >= state->filled) {
            return TINY_EXPIRED;
        }
        if (replace_input(state, updates)) {
            uint32_t age = updates[0].age;
#if TINY_ACC_FRAMES > 0
            if (mode == TINY_DELTA || mode == TINY_DELTA_ALL || mode == TINY_DELTA_NO_EQUAL) {
                delta_short_single(state, age, mode);
                return TINY_OK;
            }
#endif
            repair_recent(state, age, mode != TINY_DELTA_NO_EQUAL);
        }
        return TINY_OK;
    }
#endif
    uint64_t seen = 0;
    for (size_t index = 0; index < count; ++index) {
        const unsigned age = updates[index].age;
        if (age >= state->filled) {
            return TINY_EXPIRED;
        }
        const uint64_t bit = UINT64_C(1) << age;
        if (seen & bit) {
            return TINY_DUPLICATE;
        }
        seen |= bit;
    }

#ifdef TINY_EXTENDED
    if (mode == TINY_BATCH_SERIAL) {
        for (size_t index = 0; index < count; ++index) {
            apply_updates(state, updates + index, 1, TINY_RECENT_REPLAY);
        }
        return TINY_OK;
    }
#endif
    if (mode == TINY_SERIAL) {
        for (size_t index = 0; index < count; ++index) {
            apply_updates(state, updates + index, 1, TINY_ADAPTIVE);
        }
    } else {
        apply_updates(state, updates, count, mode);
    }
    return TINY_OK;
}
