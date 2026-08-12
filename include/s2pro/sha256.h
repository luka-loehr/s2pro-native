/* Minimal streaming SHA-256 used for cache provenance fingerprints. */
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    uint8_t  block[64];
    size_t   block_len;
} s2p_sha256;

void s2p_sha256_init(s2p_sha256* ctx);
void s2p_sha256_update(s2p_sha256* ctx, const void* data, size_t len);
void s2p_sha256_final(s2p_sha256* ctx, uint8_t out[32]);
int  s2p_sha256_file(const char* path, uint8_t out[32]);
