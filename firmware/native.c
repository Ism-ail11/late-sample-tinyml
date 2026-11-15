#include "tinyml.h"

void *tiny_create(void);
void tiny_destroy(void *ptr);
size_t tiny_state_size(void);
void tiny_reset_counts(void);
void tiny_read_counts(uint32_t *out);

#include <stdlib.h>

void *tiny_create(void)
{
    TinyState *state = malloc(sizeof(TinyState));
    if (state) {
        tiny_reset(state);
    }
    return state;
}

void tiny_destroy(void *ptr)
{
    free(ptr);
}

size_t tiny_state_size(void)
{
    return sizeof(TinyState);
}

void tiny_reset_counts(void)
{
    for (unsigned layer = 0; layer < TINY_LAYERS; ++layer) {
        tiny_layer_calls[layer] = 0;
    }
}

void tiny_read_counts(uint32_t *out)
{
    for (unsigned layer = 0; layer < TINY_LAYERS; ++layer) {
        out[layer] = tiny_layer_calls[layer];
    }
}

void tiny_native_case(const int8_t *provisional, unsigned ticks, const TinyUpdate *updates,
                      unsigned count, unsigned mode, int8_t *before, int8_t *after, uint32_t *work)
{
    TinyState state;
    tiny_reset(&state);
    for (unsigned tick = 0; tick < ticks; ++tick) {
        tiny_append(&state, provisional + tick * TINY_INPUTS);
    }
    for (unsigned age = 0; age < state.filled; ++age) {
        const int8_t *scores = tiny_scores(&state, (uint8_t)age);
        for (unsigned i = 0; i < TINY_CLASSES; ++i) {
            before[age * TINY_CLASSES + i] = scores[i];
        }
    }
    tiny_reset_counts();
    TinyStatus status = tiny_repair(&state, updates, count, (TinyMode)mode);
    tiny_read_counts(work);
    work[TINY_LAYERS] = status;
    for (unsigned age = 0; age < state.filled; ++age) {
        const int8_t *scores = tiny_scores(&state, (uint8_t)age);
        for (unsigned i = 0; i < TINY_CLASSES; ++i) {
            after[age * TINY_CLASSES + i] = scores[i];
        }
    }
}
