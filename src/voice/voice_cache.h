#pragma once

#include <stdint.h>

int s2p_voice_cache_load(const char* cache_path, const char* wav_path,
                         const uint8_t producer[32], int32_t** codes_out,
                         int* frames_out);
int s2p_voice_cache_store(const char* cache_path, const char* wav_path,
                          const uint8_t producer[32], const int32_t* codes,
                          int frames);
