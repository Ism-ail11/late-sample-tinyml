#ifndef LATE_SAMPLE_TINYML_H
#define LATE_SAMPLE_TINYML_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "model_config.h"

#ifndef TINY_HORIZON
#define TINY_HORIZON 48
#endif

#define TINY_INPUTS 3
#define TINY_LAYERS 4
#define TINY_MAX_REPAIR (TINY_HORIZON + 1)
#define TINY_RING0 (TINY_MAX_REPAIR + 2 * TINY_D0)
#define TINY_RING1 (TINY_MAX_REPAIR + 2 * TINY_D1)
#define TINY_RING2 (TINY_MAX_REPAIR + 2 * TINY_D2)
#define TINY_RING3 TINY_MAX_REPAIR
#define TINY_RING4 TINY_MAX_REPAIR

#ifdef TINY_EXTENDED
#ifndef TINY_ACC_FRAMES
#define TINY_ACC_FRAMES 8
#endif
#define TINY_DELTA_FRAMES 8
#define TINY_DELTA_SHORT_FRAMES 8
#define TINY_MAX_WIDTH (TINY_C0 > TINY_C1 ? TINY_C0 : TINY_C1)
#if TINY_ACC_FRAMES > 0
_Static_assert(TINY_ACC_FRAMES < 64 && (TINY_ACC_FRAMES & (TINY_ACC_FRAMES - 1)) == 0,
               "Accumulator ring capacity must be a power of two below 64.");
#endif
#endif

_Static_assert(TINY_HORIZON >= 0 && TINY_HORIZON <= 63, "The dirty mask supports ages 0 through 63.");

typedef struct {
    uint8_t age;
    int8_t sample[TINY_INPUTS];
} TinyUpdate;

typedef struct {
    uint16_t head[TINY_LAYERS + 1];
    uint8_t filled;
    int8_t input[TINY_RING0][TINY_INPUTS];
    int8_t hidden0[TINY_RING1][TINY_C0];
    int8_t hidden1[TINY_RING2][TINY_C1];
    int8_t hidden2[TINY_RING3][TINY_C2];
    int8_t scores[TINY_RING4][TINY_CLASSES];
#if defined(TINY_EXTENDED) && TINY_ACC_FRAMES > 0
    uint8_t accumulator_head;
    int32_t acc1[TINY_ACC_FRAMES][TINY_C1];
    int32_t acc2[TINY_ACC_FRAMES][TINY_C2];
#endif
} TinyState;

typedef enum {
    TINY_REPLAY = 0,
    TINY_SPAN = 1,
    TINY_TAP_ONLY = 2,
    TINY_TIME_EQUAL = 3,
    TINY_SPAN_EQUAL = 4,
    TINY_ADAPTIVE = 5,
    TINY_SERIAL = 6,
    TINY_ADAPTIVE_NO_EQUAL = 7,
#ifdef TINY_EXTENDED
    TINY_AGE_MASK = 8,
    TINY_AGE_EQUAL = 9,
    TINY_DIRECT_REPLAY = 10,
    TINY_DELTA = 11,
    TINY_DELTA_GENERIC = 12,
    TINY_DELTA_DISABLED = 13,
    TINY_DELTA_ALL = 14,
    TINY_DELTA_NO_EQUAL = 15,
    TINY_BATCH_SERIAL = 16,
    TINY_RECENT_REPLAY = 17,
#endif
    TINY_MODE_COUNT
} TinyMode;

typedef enum {
    TINY_OK = 0,
    TINY_EXPIRED = 1,
    TINY_DUPLICATE = 2,
    TINY_INVALID = 3,
} TinyStatus;

void tiny_reset(TinyState *state);
void tiny_append(TinyState *state, const int8_t sample[TINY_INPUTS]);
const int8_t *tiny_scores(const TinyState *state, uint8_t age);
TinyStatus tiny_repair(TinyState *state, const TinyUpdate *updates, size_t count, TinyMode mode);

#ifdef TINY_COUNT_CALLS
extern uint32_t tiny_layer_calls[TINY_LAYERS];
#endif

#endif
