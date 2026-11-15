/* Cortex-M benchmark loader and instruction counter for Unicorn 2.1.x. */
#include <elf.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unicorn/unicorn.h>
#include <unicorn/arm.h>

typedef struct {
    uint64_t begin[2], end[2];
    uint64_t before[2], after[2];
    uint64_t finished, tally;
    int phase, done, failed;
} Markers;

static int checked(uc_err error, const char *action)
{
    if (error == UC_ERR_OK) {
        return 1;
    }
    fprintf(stderr, "%s: %s\n", action, uc_strerror(error));
    return 0;
}

static void instruction(uc_engine *cpu, uint64_t address, uint32_t size, void *opaque)
{
    (void)size;
    Markers *m = opaque;
    for (int i = 0; i < 2; ++i) {
        if (address == m->begin[i]) {
            m->phase = i + 1;
            m->tally = 0;
            const uint32_t zero = 0;
            if (!checked(uc_mem_write(cpu, m->before[i], &zero, sizeof(zero)), "write start counter")) {
                m->failed = 1;
                uc_emu_stop(cpu);
            }
        } else if (address == m->end[i]) {
            if (m->phase != i + 1 || m->tally > UINT32_MAX) {
                m->failed = 1;
                uc_emu_stop(cpu);
                return;
            }
            const uint32_t elapsed = (uint32_t)m->tally;
            if (!checked(uc_mem_write(cpu, m->after[i], &elapsed, sizeof(elapsed)), "write end counter")) {
                m->failed = 1;
                uc_emu_stop(cpu);
            }
            m->phase = 0;
            return;
        }
    }
    if (address == m->finished) {
        m->done = 1;
        uc_emu_stop(cpu);
        return;
    }
    m->tally += m->phase != 0;
}

static unsigned char *read_file(const char *name, size_t *size)
{
    FILE *source = fopen(name, "rb");
    if (!source || fseek(source, 0, SEEK_END) != 0) {
        if (source) {
            fclose(source);
        }
        return NULL;
    }
    const long length = ftell(source);
    if (length < 0 || fseek(source, 0, SEEK_SET) != 0) {
        fclose(source);
        return NULL;
    }
    unsigned char *bytes = malloc((size_t)length);
    if (!bytes || fread(bytes, 1, (size_t)length, source) != (size_t)length) {
        free(bytes);
        fclose(source);
        return NULL;
    }
    fclose(source);
    *size = (size_t)length;
    return bytes;
}

static int load_elf(uc_engine *cpu, const char *name)
{
    size_t size;
    unsigned char *file = read_file(name, &size);
    if (!file) {
        perror(name);
        return 0;
    }
    int ok = 0;
    if (size < sizeof(Elf32_Ehdr)) {
        goto finish;
    }
    Elf32_Ehdr header;
    memcpy(&header, file, sizeof(header));
    if (memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS32 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_machine != EM_ARM ||
        header.e_phentsize != sizeof(Elf32_Phdr) || header.e_phoff > size ||
        (size_t)header.e_phnum * sizeof(Elf32_Phdr) > size - header.e_phoff) {
        goto finish;
    }
    for (unsigned i = 0; i < header.e_phnum; ++i) {
        Elf32_Phdr segment;
        memcpy(&segment, file + header.e_phoff + i * sizeof(segment), sizeof(segment));
        if (segment.p_type != PT_LOAD || segment.p_filesz == 0) {
            continue;
        }
        if (segment.p_offset > size || segment.p_filesz > size - segment.p_offset ||
            !checked(uc_mem_write(cpu, segment.p_paddr, file + segment.p_offset, segment.p_filesz),
                     "load ELF segment")) {
            goto finish;
        }
    }
    ok = 1;
finish:
    free(file);
    if (!ok) {
        fprintf(stderr, "Invalid ARM ELF image: %s\n", name);
    }
    return ok;
}

static int emit_memory(uc_engine *cpu, uint64_t address, size_t size, const char *path)
{
    unsigned char *bytes = malloc(size);
    if (!bytes || !checked(uc_mem_read(cpu, address, bytes, size), "read result buffer")) {
        free(bytes);
        return 0;
    }
    FILE *handle = fopen(path, "wb");
    if (!handle) {
        perror(path);
        free(bytes);
        return 0;
    }
    int ok = (fwrite(bytes, 1, size, handle) == size);
    ok = (fclose(handle) == 0) && ok;
    free(bytes);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 17) {
        fprintf(stderr, "Usage: cpu_runner elf cpu repair-begin repair-end repair-start-count repair-end-count "
                        "append-begin append-end append-start-count append-end-count done result-addr result-size "
                        "results state-addr state-output\n");
        return 2;
    }
    const int model = !strcmp(argv[2], "cortex-m4") ? UC_CPU_ARM_CORTEX_M4 :
                      !strcmp(argv[2], "cortex-m33") ? UC_CPU_ARM_CORTEX_M33 : -1;
    if (model < 0) {
        fputs("Unsupported CPU model\n", stderr);
        return 2;
    }
    Markers markers = {0};
    for (int i = 0; i < 2; ++i) {
        markers.begin[i] = strtoull(argv[3 + 4 * i], NULL, 0);
        markers.end[i] = strtoull(argv[4 + 4 * i], NULL, 0);
        markers.before[i] = strtoull(argv[5 + 4 * i], NULL, 0);
        markers.after[i] = strtoull(argv[6 + 4 * i], NULL, 0);
    }
    markers.finished = strtoull(argv[11], NULL, 0);
    const uint64_t result_addr = strtoull(argv[12], NULL, 0);
    const size_t result_size = strtoul(argv[13], NULL, 0);
    const uint64_t state_addr = strtoull(argv[15], NULL, 0);
    uc_engine *cpu;
    if (!checked(uc_open(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS, &cpu), "open ARM engine")) {
        return 1;
    }
    int success = 0;
    if (!checked(uc_ctl_set_cpu_model(cpu, model), "set CPU") ||
        !checked(uc_mem_map(cpu, 0x08000000, 0x100000, UC_PROT_ALL), "map flash") ||
        !checked(uc_mem_map(cpu, 0x20000000, 0x20000, UC_PROT_ALL), "map SRAM") ||
        !load_elf(cpu, argv[1])) {
        goto finish;
    }
    uint32_t vectors[2];
    uc_hook hook;
    if (!checked(uc_mem_read(cpu, 0x08000000, vectors, sizeof(vectors)), "read vectors") ||
        !checked(uc_reg_write(cpu, UC_ARM_REG_SP, &vectors[0]), "set stack pointer") ||
        !checked(uc_hook_add(cpu, &hook, UC_HOOK_CODE, instruction, &markers, 1, 0), "install instruction hook") ||
        !checked(uc_emu_start(cpu, vectors[1], 0, 0, 0), "execute ARM firmware")) {
        goto finish;
    }
    if (!markers.done || markers.failed) {
        fputs("Firmware did not reach the completion marker\n", stderr);
        goto finish;
    }
    success = emit_memory(cpu, result_addr, result_size, argv[14]) &&
              emit_memory(cpu, state_addr, sizeof(uint32_t), argv[16]);
finish:
    uc_close(cpu);
    return success ? 0 : 1;
}
