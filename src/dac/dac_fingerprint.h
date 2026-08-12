#pragma once

#include <stdint.h>

#include "s2pro/dac.h"

/* Stable identity of the loaded codec artifact plus encoder implementation
 * and precision policy. Voice-code caches must match it byte for byte. */
void s2p_dac_encoder_fingerprint(const s2p_dac* d, uint8_t out[32]);
