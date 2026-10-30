/* Standalone memory/UB validation of the exported C interface. */
#include "../csrc/sampen.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

int main(void) {
    enum { channels = 6, samples = 513 };
    double *x = malloc(channels*samples*sizeof(double));
    assert(x);
    double radii[channels], tree[channels], sweep[channels], parallel[channels], range[channels],bitset[channels];
    uint64_t b1[channels],a1[channels],b2[channels],a2[channels];
    for (size_t ch=0; ch<channels; ++ch) {
        for (size_t i=0; i<samples; ++i)
            x[ch*samples+i] = ch == 0 ? 1.0 : ch == 1 ? (double)(i%4) :
                                      sin((double)i*.17+(double)ch)*cos((double)i*.07);
    }
    for (size_t order=2; order<=6; ++order) {
        for (size_t delay=1; delay<=9; ++delay) {
            for (int radius=0; radius<4; ++radius) {
                for (size_t ch=0; ch<channels; ++ch) radii[ch] = (double)radius*.3;
                assert(sampen_channels(x,channels,samples,order,delay,radii,1,0,tree)==0);
                assert(sampen_channels(x,channels,samples,order,delay,radii,1,1,sweep)==0);
                assert(sampen_channels(x,channels,samples,order,delay,radii,4,0,parallel)==0);
                if (order==2) assert(sampen_channels(x,channels,samples,order,delay,radii,4,2,range)==0);
                assert(sampen_channels_counts(x,channels,samples,order,delay,radii,1,0,tree,b1,a1)==0);
                assert(sampen_channels_counts(x,channels,samples,order,delay,radii,4,3,bitset,b2,a2)==0);
                for (size_t ch=0; ch<channels; ++ch) {
                    assert(b1[ch]==b2[ch] && a1[ch]==a2[ch]);
                    assert((isnan(tree[ch]) && isnan(bitset[ch])) || tree[ch]==bitset[ch]);
                    if (order==2) assert((isnan(tree[ch]) && isnan(range[ch])) || tree[ch] == range[ch]);
                    assert((isnan(tree[ch]) && isnan(sweep[ch])) || tree[ch] == sweep[ch]);
                    assert((isnan(tree[ch]) && isnan(parallel[ch])) || tree[ch] == parallel[ch]);
                }
            }
        }
    }
    assert(sampen_channels(x,channels,samples,SIZE_MAX,2,radii,1,0,tree)==2);
    assert(sampen_channels(x,channels,samples,2,0,radii,1,0,tree)==2);
    assert(sampen_channels(NULL,channels,samples,2,1,radii,1,0,tree)==2);
    free(x);
    return 0;
}
