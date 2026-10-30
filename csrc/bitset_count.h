/* Included twice to provide a portable counter and runtime-dispatched POPCNT.
   Macros are local to sampen.c; both variants implement the same integer count. */
COUNT_TARGET static MatchCounts COUNT_NAME(size_t n, size_t first, size_t end, size_t order,
        size_t words, const uint64_t **low, const uint64_t **high,
        const uint64_t *adjust) {
    MatchCounts result={0,0};
    size_t begin=first/64, last=(end-1)/64;
    uint64_t last_mask=UINT64_MAX >> (63-((end-1)%64));
    uint64_t first_mask=UINT64_MAX << (first%64);
    if (order==3) {
        for (size_t w=begin; w<=last; ++w) {
            uint64_t b=(low[0][w]^high[0][w]^adjust[w]) &
                       (low[1][w]^high[1][w]^adjust[words+w]);
            if(w==begin) b&=first_mask;
            if(w==last) b&=last_mask;
            if(!b) continue;
            result.b+=COUNT_POP(b);
            result.a+=COUNT_POP(b & (low[2][w]^high[2][w]^adjust[2*words+w]));
        }
    } else {
        for (size_t w=begin; w<=last; ++w) {
            uint64_t b=w==begin ? first_mask : UINT64_MAX;
            for(size_t k=0;k+1<order && b;++k)
                b&=low[k][w]^high[k][w]^adjust[k*words+w];
            if(w==last) b&=last_mask;
            if(!b) continue;
            result.b+=COUNT_POP(b);
            result.a+=COUNT_POP(b & (low[order-1][w]^high[order-1][w]^adjust[(order-1)*words+w]));
        }
    }
    (void)n;
    return result;
}
