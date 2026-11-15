#include "tinyml.h"

#define TRIAL_STACK_BYTES 1024u
#define CANARY UINT32_C(0x9326fcd1)

typedef struct {
    const int8_t (*provisional)[TINY_INPUTS];
    const TinyUpdate *updates;
    const int8_t *before;
    const int8_t *after;
    uint16_t ticks;
    uint8_t count;
    uint8_t expected_status;
} BenchCase;

#include "bench_cases.h"

#define CASE_COUNT (sizeof(bench_cases) / sizeof(bench_cases[0]))
#ifdef TINY_BENCH_SUBSET
#define BENCH_MODE_COUNT (sizeof(bench_modes) / sizeof(bench_modes[0]))
#else
#define BENCH_MODE_COUNT TINY_MODE_COUNT
#endif
#define RESULT_COUNT (CASE_COUNT * BENCH_MODE_COUNT)

static TinyState state;
static TinyState provisional_state;
static uint32_t trial_stack[TRIAL_STACK_BYTES / sizeof(uint32_t)] __attribute__((aligned(8), used));

/* Renode writes the counter snapshots via the memory bus. */
volatile uint32_t bench_begin_count;
volatile uint32_t bench_end_count;
#ifdef TINY_EXTENDED
volatile uint32_t bench_append_begin_count;
volatile uint32_t bench_append_end_count;
#endif

typedef struct {
    uint32_t instructions;
    uint32_t mismatches;
    uint32_t stack_bytes;
    uint32_t checksum;
#ifdef TINY_EXTENDED
    uint32_t append_instructions;
#endif
} BenchResult;

/* Retain all results for one bulk export at benchmark_finished. */
volatile BenchResult bench_results[RESULT_COUNT];
volatile uint32_t bench_state_bytes;

__attribute__((noipa))
void correction_begin(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noipa))
void correction_end(void)
{
    __asm__ volatile("" ::: "memory");
}

#ifdef TINY_EXTENDED
__attribute__((noipa))
void append_begin(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noipa))
void append_end(void)
{
    __asm__ volatile("" ::: "memory");
}
#endif

__attribute__((noipa))
void benchmark_finished(void)
{
    __asm__ volatile("" ::: "memory");
}

/* Keep the harness stack outside the watermark. The AAPCS preserves r4 across the call. */
__attribute__((naked, noipa))
static TinyStatus measured_repair(TinyState *target __attribute__((unused)),
                                  const TinyUpdate *updates __attribute__((unused)),
                                  size_t count __attribute__((unused)),
                                  TinyMode mode __attribute__((unused)))
{
    __asm__ volatile(
        "push {r4, lr}\n"
        "mov r4, sp\n"
        "ldr r12, =trial_stack + 1024\n"
        "mov sp, r12\n"
        "bl tiny_repair\n"
        "mov sp, r4\n"
        "pop {r4, pc}\n"
    );
}

static uint32_t check_scores(const int8_t *expected)
{
    uint32_t failures = 0;
    for (unsigned age = 0; age < TINY_MAX_REPAIR; ++age) {
        const int8_t *result = tiny_scores(&state, (uint8_t)age);
        for (unsigned label = 0; label < TINY_CLASSES; ++label) {
            failures += result[label] != expected[age * TINY_CLASSES + label];
        }
    }
    return failures;
}

int main(void)
{
    bench_state_bytes = sizeof(state);
    for (unsigned test = 0; test < CASE_COUNT; ++test) {
        const BenchCase *input = &bench_cases[test];
        tiny_reset(&provisional_state);
#ifdef TINY_EXTENDED
        for (unsigned tick = 0; tick + 1u < input->ticks; ++tick) {
            tiny_append(&provisional_state, input->provisional[tick]);
        }
        append_begin();
        tiny_append(&provisional_state, input->provisional[input->ticks - 1u]);
        append_end();
#else
        for (unsigned tick = 0; tick < input->ticks; ++tick) {
            tiny_append(&provisional_state, input->provisional[tick]);
        }
#endif
        for (unsigned mode = 0; mode < BENCH_MODE_COUNT; ++mode) {
#ifdef TINY_BENCH_SUBSET
            const TinyMode selected_mode = bench_modes[mode];
#else
            const TinyMode selected_mode = (TinyMode)mode;
#endif
            unsigned char *destination = (unsigned char *)&state;
            const unsigned char *source = (const unsigned char *)&provisional_state;
            for (size_t i = 0; i < sizeof(state); ++i) {
                destination[i] = source[i];
            }
            uint32_t wrong = check_scores(input->before);
            for (unsigned word = 0; word < sizeof(trial_stack) / sizeof(trial_stack[0]); ++word) {
                trial_stack[word] = CANARY;
            }
            correction_begin();
            TinyStatus status = measured_repair(&state, input->updates, input->count, selected_mode);
            correction_end();
            unsigned unused = 0;
            while (unused < sizeof(trial_stack) / sizeof(trial_stack[0]) && trial_stack[unused] == CANARY) {
                ++unused;
            }
            wrong += status != (TinyStatus)input->expected_status;
            wrong += check_scores(input->after);
            const int8_t *latest = tiny_scores(&state, 0);
            uint32_t checksum = 0;
            for (unsigned label = 0; label < TINY_CLASSES; ++label) {
                checksum = 31u * checksum + (uint8_t)latest[label];
            }
            volatile BenchResult *result = &bench_results[test * BENCH_MODE_COUNT + mode];
            result->instructions = bench_end_count - bench_begin_count;
            result->mismatches = wrong;
            result->stack_bytes = sizeof(trial_stack) - unused * sizeof(trial_stack[0]);
            result->checksum = checksum;
#ifdef TINY_EXTENDED
            result->append_instructions = bench_append_end_count - bench_append_begin_count;
#endif
        }
    }
    benchmark_finished();
    return 0;
}
