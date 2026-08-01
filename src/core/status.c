/* s2pro-native — status code strings. */
#include "s2pro/status.h"

const char* s2p_status_str(s2p_status s)
{
    switch (s) {
    case S2P_OK:              return "ok";
    case S2P_ERR_INVALID:     return "invalid argument";
    case S2P_ERR_CUDA:        return "cuda error";
    case S2P_ERR_IO:          return "i/o error";
    case S2P_ERR_OOM:         return "out of memory";
    case S2P_ERR_FORMAT:      return "format/parse error";
    case S2P_ERR_STATE:       return "invalid state";
    case S2P_ERR_UNSUPPORTED: return "unsupported";
    case S2P_ERR_INTERNAL:    return "internal invariant violated";
    case S2P_ERR_FULL:        return "capacity exhausted";
    }
    return "unknown status";
}
