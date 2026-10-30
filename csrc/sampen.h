#ifndef SAMPEN_H
#define SAMPEN_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Input: contiguous row-major channels x samples, finite doubles. Radii and
   output each contain channels doubles. Caller owns all buffers. No self
   matches; inclusive Chebyshev radius; both dimensions use samples-order*delay
   starts. Zero matches produce NaN. Thread-safe, no persistent C workspaces.
   algorithm: 0 = dual tree, 1 = sorted sweep, 2 = range counter (order=2 only).
   status: 0 = success, 1 = allocation failure, 2 = invalid input/size overflow.
   A failure can leave some output channels completed; discard all on nonzero. */
int sampen_channels(const double *data, size_t channels, size_t samples,
                    size_t order, size_t delay, const double *radii,
                    int threads, int algorithm, double *out);
int sampen_has_openmp(void);
#ifdef __cplusplus
}
#endif
#endif
