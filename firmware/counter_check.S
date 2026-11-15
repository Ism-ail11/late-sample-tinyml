/* Counter self-test with a known dynamic Thumb instruction count. */
.syntax unified
.thumb

.section .isr_vector, "a", %progbits
.word _estack
.word Reset_Handler

.text
.global Reset_Handler
.global counter_begin
.global counter_end
.type Reset_Handler, %function
.thumb_func
Reset_Handler:
    b counter_begin

.type counter_begin, %function
.thumb_func
counter_begin:
    movs r0, #4
1:
    subs r0, #1
    bne 1b
    nop
    b counter_end

.type counter_end, %function
.thumb_func
counter_end:
    wfi
    b counter_end
