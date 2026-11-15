#ifndef TINY_SIMD_H
#define TINY_SIMD_H

#include <stdint.h>

static inline int32_t tiny_dot(int32_t sum, const int8_t *a, const int8_t *b, unsigned length)
{
    unsigned i = 0;
#if defined(__ARM_FEATURE_DSP) && defined(__ARMEL__) && !defined(TINY_SCALAR_DOT)
    for (; i + 4 <= length; i += 4) {
        uint32_t left;
        uint32_t right;
        uint32_t l0, r0, l1, r1;
        __builtin_memcpy(&left, a + i, sizeof(left));
        __builtin_memcpy(&right, b + i, sizeof(right));
        __asm__("sxtb16 %0, %4\n"
                "sxtb16 %1, %5\n"
                "sxtb16 %2, %4, ror #8\n"
                "sxtb16 %3, %5, ror #8"
                : "=&r"(l0), "=&r"(r0), "=&r"(l1), "=&r"(r1) : "r"(left), "r"(right));
        __asm__("smlad %0, %1, %2, %0" : "+r"(sum) : "r"(l0), "r"(r0));
        __asm__("smlad %0, %1, %2, %0" : "+r"(sum) : "r"(l1), "r"(r1));
    }
#endif
    for (; i < length; ++i) {
        sum += (int32_t)a[i] * b[i];
    }
    return sum;
}

#endif
