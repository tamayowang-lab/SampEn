/* Runtime-dispatched AVX2 path. Ranks, not amplitudes, are represented in 16
   bits. Exact floating-point radius bounds were computed before this step. */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(SAMPEN_NO_POPCNT) && !defined(SAMPEN_NO_SIMD)
#include <immintrin.h>

__attribute__((target("avx2"),always_inline))
static inline uint64_t rank_neighborhood(const int16_t *ranks,uint32_t lo,uint32_t hi) {
    /* Stored rank is p-32768. Wrapped subtraction by lo maps unsigned
       (p-lo) into signed order, so one comparison tests the whole interval.
       Width 65536 maps the limit to 32767 and correctly accepts all ranks. */
    short low_bits=(short)(lo<=32767 ? (int)lo : (int)lo-65536);
    __m256i low=_mm256_set1_epi16(low_bits);
    __m256i limit=_mm256_set1_epi16((short)((int)(hi-lo)-1-32768));
    uint64_t mask=0;
    for(int half=0;half<2;++half) {
        __m256i a=_mm256_loadu_si256((const __m256i*)(ranks+32*half));
        __m256i b=_mm256_loadu_si256((const __m256i*)(ranks+32*half+16));
        __m256i outside_a=_mm256_cmpgt_epi16(_mm256_sub_epi16(a,low),limit);
        __m256i outside_b=_mm256_cmpgt_epi16(_mm256_sub_epi16(b,low),limit);
        __m256i bytes=_mm256_packs_epi16(outside_a,outside_b);
        bytes=_mm256_permute4x64_epi64(bytes,0xd8);
        uint32_t inside=~(uint32_t)_mm256_movemask_epi8(bytes);
        mask|=(uint64_t)inside<<(32*half);
    }
    return mask;
}

__attribute__((target("avx2,popcnt")))
static MatchCounts count_rank_masks(size_t first,size_t end,size_t padded,
        const int16_t *ranks,const uint32_t *lower,const uint32_t *upper) {
    MatchCounts result={0,0};
    size_t begin=first/64,last=(end-1)/64;
    for(size_t w=begin;w<=last;++w) {
        uint64_t b=rank_neighborhood(ranks+64*w,lower[0],upper[0]) &
                   rank_neighborhood(ranks+padded+64*w,lower[1],upper[1]);
        if(w==begin) b&=UINT64_MAX << (first%64);
        if(w==last) b&=UINT64_MAX >> (63-((end-1)%64));
        if(!b) continue;
        result.b+=(uint64_t)__builtin_popcountll(b);
        result.a+=(uint64_t)__builtin_popcountll(b & rank_neighborhood(ranks+2*padded+64*w,lower[2],upper[2]));
    }
    return result;
}

static int entropy_rank(const double *x,size_t n,size_t delay,double r,
        double *result,uint64_t *b_out,uint64_t *a_out) {
    size_t padded=(n/64+(n%64!=0))*64;
    uint32_t *first_hi=malloc(n*sizeof(uint32_t));
    uint32_t *lower=malloc(3*n*sizeof(uint32_t)),*upper=malloc(3*n*sizeof(uint32_t));
    int16_t *ranks=calloc(3*padded,sizeof(int16_t));
    int error=1;
    if(!first_hi || !lower || !upper || !ranks) goto cleanup;
    error=prepare_radius_views(x,n,3,delay,r,first_hi,lower,upper,NULL,ranks,padded);
    if(error) goto cleanup;
    uint64_t b=0,a=0;
    for(size_t i=0;i+1<n;++i) {
        size_t end=first_hi[i];
        if(end<=i+1) continue;
        uint32_t lo[3]={lower[i],lower[n+i],lower[2*n+i]};
        uint32_t hi[3]={upper[i],upper[n+i],upper[2*n+i]};
        MatchCounts counts=count_rank_masks(i+1,end,padded,ranks,lo,hi);
        b+=counts.b;a+=counts.a;
    }
    if(b_out) *b_out=b;
    if(a_out) *a_out=a;
    *result=(!a || !b) ? NAN : -log((double)a/(double)b);
cleanup:
    free(first_hi);free(lower);free(upper);free(ranks);
    return error;
}
#endif
