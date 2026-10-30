/* Exact, aligned Chebyshev sample entropy. No approximate search/fast math. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include "sampen.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double value; size_t index; } Entry;
typedef struct { size_t begin, end, left, right; } Node;
typedef struct {
    size_t n, dim, used;
    double *points, *bounds;
    size_t *indices;
    Node *nodes;
} Tree;

int sampen_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

static int compare_entry(const void *a, const void *b) {
    double x = ((const Entry *)a)->value, y = ((const Entry *)b)->value;
    return (x > y) - (x < y);
}

static void swap_index(size_t *a, size_t *b) {
    size_t temp = *a; *a = *b; *b = temp;
}

/* Partition indices around an exact median without changing sample values. */
static void select_median(Tree *t, size_t lo, size_t hi, size_t kth, size_t axis) {
    while (hi - lo > 1) {
        double pivot = t->points[t->indices[lo + (hi-lo)/2]*t->dim + axis];
        size_t i = lo, j = hi;
        for (;;) {
            while (i < hi && t->points[t->indices[i]*t->dim + axis] < pivot) ++i;
            while (j > lo && t->points[t->indices[j-1]*t->dim + axis] > pivot) --j;
            if (i >= j) break;
            swap_index(&t->indices[i], &t->indices[j-1]);
            ++i; --j;
        }
        if (kth < j) hi = j;
        else if (kth >= i) lo = i;
        else return;
    }
}

static size_t build_tree(Tree *t, size_t begin, size_t end) {
    size_t id = t->used++;
    Node *node = &t->nodes[id];
    node->begin = begin; node->end = end;
    node->left = node->right = SIZE_MAX;
    double *low = t->bounds + id*2*t->dim;
    double *high = low + t->dim;
    for (size_t k = 0; k < t->dim; ++k) {
        low[k] = INFINITY; high[k] = -INFINITY;
    }
    for (size_t p = begin; p < end; ++p) {
        const double *point = t->points + t->indices[p]*t->dim;
        for (size_t k = 0; k < t->dim; ++k) {
            if (point[k] < low[k]) low[k] = point[k];
            if (point[k] > high[k]) high[k] = point[k];
        }
    }
    if (end - begin <= 16) return id;
    size_t axis = 0;
    for (size_t k = 1; k < t->dim; ++k)
        if (high[k] - low[k] > high[axis] - low[axis]) axis = k;
    size_t mid = begin + (end-begin)/2;
    select_median(t, begin, end, mid, axis);
    node->left = build_tree(t, begin, mid);
    node->right = build_tree(t, mid, end);
    return id;
}

/* Count unordered pairs of boxes instead of querying once per template.
   Fully matching boxes contribute a Cartesian-product count immediately. */
static void count_pairs(const Tree *t, size_t aid, size_t bid, double r,
                        int need_b, int need_a, uint64_t *b, uint64_t *a) {
    const Node *an = &t->nodes[aid], *bn = &t->nodes[bid];
    const double *al = t->bounds + aid*2*t->dim, *ah = al+t->dim;
    const double *bl = t->bounds + bid*2*t->dim, *bh = bl+t->dim;
    int inside_b = 1;
    size_t last = t->dim-1;
    if (need_b) {
        for (size_t k = 0; k < last; ++k) {
            if ((al[k] > bh[k] && al[k]-bh[k] > r) ||
                (bl[k] > ah[k] && bl[k]-ah[k] > r)) return;
            if (fabs(al[k]-bh[k]) > r || fabs(ah[k]-bl[k]) > r) inside_b = 0;
        }
    }
    uint64_t na = an->end-an->begin, nb = bn->end-bn->begin;
    uint64_t pairs = aid == bid ? na*(na-1)/2 : na*nb;
    if (need_b && inside_b) { *b += pairs; need_b = 0; }
    if (need_a) {
        if ((al[last] > bh[last] && al[last]-bh[last] > r) ||
            (bl[last] > ah[last] && bl[last]-ah[last] > r)) need_a = 0;
        else if (inside_b && fabs(al[last]-bh[last]) <= r && fabs(ah[last]-bl[last]) <= r) {
            *a += pairs; need_a = 0;
        }
    }
    if (!need_b && !need_a) return;
    if (aid == bid && an->left != SIZE_MAX) {
        count_pairs(t, an->left, an->left, r, need_b, need_a, b, a);
        count_pairs(t, an->left, an->right, r, need_b, need_a, b, a);
        count_pairs(t, an->right, an->right, r, need_b, need_a, b, a);
        return;
    }
    if (an->left != SIZE_MAX && (bn->left == SIZE_MAX || na >= nb)) {
        count_pairs(t, an->left, bid, r, need_b, need_a, b, a);
        count_pairs(t, an->right, bid, r, need_b, need_a, b, a);
        return;
    }
    if (bn->left != SIZE_MAX) {
        count_pairs(t, aid, bn->left, r, need_b, need_a, b, a);
        count_pairs(t, aid, bn->right, r, need_b, need_a, b, a);
        return;
    }
    for (size_t i = an->begin; i < an->end; ++i) {
        const double *p = t->points+t->indices[i]*t->dim;
        size_t begin = aid == bid ? i+1 : bn->begin;
        for (size_t j = begin; j < bn->end; ++j) {
            const double *q = t->points+t->indices[j]*t->dim;
            int match = 1;
            if (!inside_b) {
                for (size_t k = 0; k < last; ++k)
                    if (fabs(p[k]-q[k]) > r) { match = 0; break; }
            }
            if (match) {
                if (need_b) ++*b;
                if (need_a && fabs(p[last]-q[last]) <= r) ++*a;
            }
        }
    }
}

static int entropy_tree(const double *x, size_t n, size_t order, size_t delay,
                        double r, double *result) {
    size_t dim = order + 1;
    /* Bound all products before allocation, including bounds' 4*n*dim. */
    if (n > SIZE_MAX / sizeof(double) / 4 / dim ||
        n > SIZE_MAX / sizeof(Node) / 2) return 2;
    Tree t = {n, dim, 0, NULL, NULL, NULL, NULL};
    t.points = malloc(n*dim*sizeof(double));
    t.bounds = malloc(4*n*dim*sizeof(double));
    t.indices = malloc(n*sizeof(size_t));
    t.nodes = malloc(2*n*sizeof(Node));
    if (!t.points || !t.bounds || !t.indices || !t.nodes) {
        free(t.points); free(t.bounds); free(t.indices); free(t.nodes); return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        t.indices[i] = i;
        for (size_t k = 0; k < dim; ++k) t.points[i*dim+k] = x[i+k*delay];
    }
    build_tree(&t, 0, n);
    uint64_t b = 0, a = 0;
    count_pairs(&t, 0, 0, r, 1, 1, &b, &a);
    *result = (!a || !b) ? NAN : -log((double)a/(double)b);
    free(t.points); free(t.bounds); free(t.indices); free(t.nodes);
    return 0;
}

static int entropy_sweep(const double *x, size_t n, size_t order, size_t delay,
                         double r, double *result) {
    if (n > SIZE_MAX/sizeof(Entry)) return 2;
    Entry *entries = malloc(n*sizeof(Entry));
    if (!entries) return 1;
    for (size_t i = 0; i < n; ++i) entries[i] = (Entry){x[i], i};
    qsort(entries, n, sizeof(Entry), compare_entry);
    uint64_t b = 0, a = 0;
    for (size_t left = 0; left + 1 < n; ++left) {
        size_t i = entries[left].index;
        for (size_t right = left+1; right < n; ++right) {
            if (entries[right].value-entries[left].value > r) break;
            size_t j = entries[right].index;
            int match = 1;
            for (size_t k = 1; k < order; ++k)
                if (fabs(x[i+k*delay]-x[j+k*delay]) > r) { match = 0; break; }
            if (match) {
                ++b;
                if (fabs(x[i+order*delay]-x[j+order*delay]) <= r) ++a;
            }
        }
    }
    *result = (!a || !b) ? NAN : -log((double)a/(double)b);
    free(entries);
    return 0;
}

/* For order=2, sweep coordinate 0 while a dynamic 2D Fenwick structure counts
   the rectangle in coordinates 1 and 2. Exact O(n log^2 n) time and O(n log n)
   workspace, even when many template pairs match. */
typedef struct {
    size_t n, *offset, *size;
    double *coordinates;
    int64_t *counts, *ycounts;
} RangeIndex;

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static size_t lower_value(const double *v, size_t n, double value) {
    size_t lo=0, hi=n;
    while (lo < hi) { size_t mid=lo+(hi-lo)/2;
        if (v[mid] < value) lo=mid+1; else hi=mid;
    }
    return lo;
}

/* Compare differences rather than rounded value +/- radius endpoints. */
static size_t radius_bound(const double *v, size_t n, double q, double r, int upper) {
    size_t lo=0, hi=n;
    while (lo < hi) { size_t mid=lo+(hi-lo)/2;
        int go_right = upper ? !(v[mid] > q && v[mid]-q > r)
                             : (q > v[mid] && q-v[mid] > r);
        if (go_right) lo=mid+1; else hi=mid;
    }
    return lo;
}

static int64_t bit_prefix(const int64_t *counts, size_t p) {
    int64_t result=0;
    for (; p; p-=p & (~p+1)) result+=counts[p];
    return result;
}

static void range_update(RangeIndex *index, size_t rank, double z, int delta) {
    for (size_t y=rank+1; y<=index->n; y+=y & (~y+1)) {
        index->ycounts[y]+=delta;
        size_t offset=index->offset[y], size=index->size[y];
        size_t p=lower_value(index->coordinates+offset,size,z)+1;
        for (; p<=size; p+=p & (~p+1)) index->counts[offset+p]+=delta;
    }
}

static int64_t rectangle_prefix(const RangeIndex *index, size_t y, double z, double r) {
    int64_t result=0;
    for (; y; y-=y & (~y+1)) {
        size_t offset=index->offset[y], size=index->size[y];
        const double *v=index->coordinates+offset;
        size_t lo=radius_bound(v,size,z,r,0), hi=radius_bound(v,size,z,r,1);
        result+=bit_prefix(index->counts+offset,hi)-bit_prefix(index->counts+offset,lo);
    }
    return result;
}

static int entropy_range(const double *x, size_t n, size_t delay, double r, double *result) {
    if (n > SIZE_MAX/sizeof(Entry) || n+2 > SIZE_MAX/sizeof(size_t)) return 2;
    Entry *entries=malloc(n*sizeof(Entry));
    double *ys=malloc(n*sizeof(double));
    size_t *ranks=malloc(n*sizeof(size_t)), *cursor=calloc(n+1,sizeof(size_t));
    RangeIndex index={n,NULL,NULL,NULL,NULL,NULL};
    index.offset=calloc(n+1,sizeof(size_t));
    index.size=calloc(n+1,sizeof(size_t));
    index.ycounts=calloc(n+1,sizeof(int64_t));
    int error=1;
    if (!entries || !ys || !ranks || !cursor || !index.offset || !index.size || !index.ycounts) goto cleanup;
    for (size_t i=0; i<n; ++i) { entries[i]=(Entry){x[i],i}; ys[i]=x[i+delay]; }
    qsort(entries,n,sizeof(Entry),compare_entry);
    qsort(ys,n,sizeof(double),compare_double);
    for (size_t i=0; i<n; ++i) {
        ranks[i]=lower_value(ys,n,x[i+delay]);
        for (size_t y=ranks[i]+1; y<=n; y+=y & (~y+1)) ++index.size[y];
    }
    size_t total=0;
    for (size_t y=1; y<=n; ++y) {
        if (total > SIZE_MAX-index.size[y]-1) { error=2; goto cleanup; }
        index.offset[y]=total;
        total+=index.size[y]+1;
    }
    if (total > SIZE_MAX/sizeof(double) || total > SIZE_MAX/sizeof(int64_t)) { error=2; goto cleanup; }
    index.coordinates=malloc(total*sizeof(double));
    index.counts=calloc(total,sizeof(int64_t));
    if (!index.coordinates || !index.counts) goto cleanup;
    for (size_t i=0; i<n; ++i)
        for (size_t y=ranks[i]+1; y<=n; y+=y & (~y+1))
            index.coordinates[index.offset[y]+cursor[y]++]=x[i+2*delay];
    for (size_t y=1; y<=n; ++y)
        qsort(index.coordinates+index.offset[y],index.size[y],sizeof(double),compare_double);
    uint64_t b=0, a=0;
    size_t left=0;
    for (size_t p=0; p<n; ++p) {
        size_t i=entries[p].index;
        while (left<p && entries[p].value-entries[left].value > r) {
            size_t j=entries[left++].index;
            range_update(&index,ranks[j],x[j+2*delay],-1);
        }
        size_t lo=radius_bound(ys,n,x[i+delay],r,0), hi=radius_bound(ys,n,x[i+delay],r,1);
        b+=(uint64_t)(bit_prefix(index.ycounts,hi)-bit_prefix(index.ycounts,lo));
        a+=(uint64_t)(rectangle_prefix(&index,hi,x[i+2*delay],r)-rectangle_prefix(&index,lo,x[i+2*delay],r));
        range_update(&index,ranks[i],x[i+2*delay],1);
    }
    *result=(!a || !b) ? NAN : -log((double)a/(double)b);
    error=0;
cleanup:
    free(entries); free(ys); free(ranks); free(cursor);
    free(index.offset); free(index.size); free(index.ycounts); free(index.coordinates); free(index.counts);
    return error;
}

/* Public ABI: input is contiguous row-major channels x samples; one radius and
   output per channel. Status: 0 success, 1 allocation, 2 invalid/overflow. */
int sampen_channels(const double *data, size_t channels, size_t samples,
                    size_t order, size_t delay, const double *radii,
                    int threads, int algorithm, double *out) {
    if (!data || !radii || !out || !channels || samples < 2 || order < 2 ||
        !delay || order > (samples-2)/delay || threads < 1 ||
        channels > SIZE_MAX/sizeof(double)/samples || (algorithm < 0 || algorithm > 2) || (algorithm == 2 && order != 2)) return 2;
    size_t n = samples - order*delay;
    /* Directed pair counts must fit uint64_t. */
    if (n > UINT64_MAX/n) return 2;
    int error = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(|:error) if(threads > 1)
#endif
    for (size_t ch = 0; ch < channels; ++ch) {
        const double *x = data + ch*samples;
        double r = radii[ch];
        int finite = isfinite(r) && r >= 0, constant = 1;
        for (size_t i = 0; i < samples; ++i) {
            if (!isfinite(x[i])) finite = 0;
            if (x[i] != x[0]) constant = 0;
        }
        if (!finite) { out[ch] = NAN; error |= 2; continue; }
        if (constant) { out[ch] = 0; continue; }
        int status;
        if (algorithm == 2) status=entropy_range(x,n,delay,r,&out[ch]);
        else if (algorithm == 0) status=entropy_tree(x,n,order,delay,r,&out[ch]);
        else status=entropy_sweep(x,n,order,delay,r,&out[ch]);
        if (status) out[ch] = NAN;
        error |= status;
    }
    return error;
}
