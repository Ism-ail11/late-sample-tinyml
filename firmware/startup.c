#include <stdint.h>

extern uint32_t _estack, _sdata, _edata, _sidata, _sbss, _ebss;
int main(void);

__attribute__((noreturn))
static void Default_Handler(void)
{
    for (;;) {
        __asm__ volatile("wfi");
    }
}

__attribute__((noreturn))
void Reset_Handler(void)
{
    const uint32_t *load = &_sidata;
    for (uint32_t *address = &_sdata; address < &_edata; ++address) {
        *address = *load++;
    }
    for (uint32_t *address = &_sbss; address < &_ebss; ++address) {
        *address = 0;
    }
    (void)main();
    Default_Handler();
}

__attribute__((section(".isr_vector"), used))
const uintptr_t vector_table[] = {
    (uintptr_t)&_estack, (uintptr_t)Reset_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    0, 0, 0, 0, (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    0, (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
};
