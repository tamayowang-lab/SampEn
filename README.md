# Sample entropy with delay

Latest: [alpha-only speed and exact-count benchmark](examples/results_alpha/README.md)
for 180-second, 250 Hz, 64-channel EEG at order 3.

`SampEn_utils.py` provides the original KDTree implementation and a new,
delay-aware Numba implementation for **Chebyshev distance**:

```python
import numpy as np
from SampEn_utils import sample_entropy_numba

x = np.load('examples/data/resting_eeg.npz', allow_pickle=False)['data'][0]
sampen = sample_entropy_numba(x, order=2, delay=4)  # tolerance = 0.2 * std(x)
print(sampen)
```

## Install and test

Tested with Python 3.12 on Linux. From this checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python build_c.py
python -m pytest -q
python examples/benchmark_resting_eeg.py
python examples/summarize_benchmark.py
```

The empirical example is already included, so benchmarks do not require network
access. MNE is needed only to recreate the excerpt from its upstream FIF file.
See [data provenance and license](examples/data/README.md) and
[measured speed comparisons](examples/results/README.md).

## Algorithm and compatibility

`sample_entropy_numba(x, order=2, delay=1, tolerance=None)` sorts template starts
by their first coordinate. A radius sweep skips pairs whose first coordinate
cannot match, then checks the remaining delayed coordinates with early rejection.
It counts unordered pairs once and does not allocate a pairwise distance matrix.
It uses linear auxiliary memory; worst-case runtime is still quadratic.
Constant signals have a direct zero-entropy fast path.

Delay is the distance **between coordinates** in a template. All possible template
starts are considered; the signal is not downsampled. For order `m` and delay
`d`, both the `m` and `m+1` template sets contain `N - m*d` starts. This makes every
shorter template extendable and gives the conventional count ratio `-log(A/B)`.
Distances equal to the radius are included, self matches are excluded, and a
zero numerator or denominator returns NaN, following the original public
function's convention. Nonfinite signals/radii, negative radii, invalid integer
parameters and fewer than two extendable templates raise `ValueError`.

The existing `sample_entropy` function remains unchanged in behavior and supports
other distance metrics. **For delay > 1 it has a template-alignment issue**: its
order-m set removes only one trailing start rather than `delay` starts. Therefore
the new result can differ from that original result. The benchmark reports both
the unchanged original function and an aligned KDTree reference; correctness is
checked against the aligned reference and independent exhaustive tests.

The original `_numba_sampen` helper is retained for comparison. It supports only
delay=1, excludes exact-radius matches and has different zero-count conventions;
it is not an equivalent reference on those boundary cases.

The first new Numba call includes compilation or loading a disk cache. Benchmark
tables exclude that warm-up and include input validation/wrapper overhead. They
report medians and quartiles from randomized method order. The original Numba
helper is compared only at delay=1. Runtime depends on signal length, order,
delay, tolerance and data distribution; a C rewrite alone would not remove those
algorithmic tradeoffs. This implementation does not silently select a backend
using an unverified fixed-length threshold.


## C backend: channels × time

The C backend computes one entropy per EEG channel, with the same correct
alignment as `sample_entropy_numba`. Build on Linux with GCC and OpenMP:

```sh
python build_c.py
```

```python
import numpy as np
from sampen_c import sample_entropy_c

# Shape: (n_channels, n_times). Each channel gets its own default radius.
eeg = np.load('examples/data/resting_eeg.npz', allow_pickle=False)['data']
sampen = sample_entropy_c(eeg, order=2, delay=4, n_jobs=4)
print(sampen)  # shape (n_channels,)
```

`tolerance=None` uses 0.2 × each channel's population standard deviation.
You can supply one scalar radius for every channel, or one radius per channel.
Input is converted to contiguous float64 and left unmodified. The C call releases
the Python GIL. `n_jobs=1` is serial; positive values parallelize independent
channels using OpenMP. `n_jobs=-1` uses CPU affinity, capped at the channel count;
use an explicit count if a container's CPU quota is smaller than its affinity.
A single channel is processed by one thread regardless of `n_jobs`.

The default `algorithm='auto'` uses an **exact range counter** for `order=2`.
It sorts the first template coordinate, sweeps a radius window, and uses dynamic
one- and two-dimensional Fenwick trees to count matches in the remaining
coordinates. This avoids enumerating matching pairs: runtime is bounded by
O(N log² N), with O(N log N) workspace per simultaneously processed channel.

For order=3 signals with 32,768–65,536 samples, the default now uses the exact
bitset method described below. Other orders/sizes use **exact dual-tree pair counting**. When two
bounding boxes are wholly within the radius, it counts all matching pairs at
once. It shares one tree for both embedding dimensions instead of building two
trees, and excludes self matches. Its workspace scales with samples × embedding
dimension; its worst-case runtime remains quadratic.

`algorithm='range'` explicitly selects the order-2 range method; `algorithm='tree'`
selects dual-tree counting for any supported order. `algorithm='sweep'` provides
a direct C counterpart of the Numba sorted-radius sweep to separate language
effects from algorithm effects. No backend uses approximate search or fast-math.
All return the same aligned result within floating-point rounding.

The shared library `_sampen_c.so` is a local build output, not committed. Rebuild
after changing C source and restart Python if the old library was already loaded. `python build_c.py --no-openmp` builds a serial version;
parallel calls then raise an informative error. Build failures preserve compiler
errors and return a nonzero exit status. Python requires only NumPy to use the
C wrapper. The standalone C interface is declared in `csrc/sampen.h`.

### Empirical channel-wise benchmark

[Batch timings and plots](examples/results_channelwise/README.md) compare C with
serial and parallel Numba and the aligned KDTree implementation. Numba's parallel
baseline uses a compiled `prange` channel loop, rather than Python threads held
back by the GIL. Both parallel baselines use the same thread count as C.
Times include validation, radius calculation, allocations and tree construction;
Numba compilation and warm-up are excluded.

Run a small offline comparison using the included excerpt:

```sh
python examples/benchmark_channelwise.py --output /tmp/sampen-quick
```

For the full long-recording/many-channel benchmark, download the pinned FIF from
[the data provenance page](examples/data/README.md), then run:

```sh
curl --fail --location \
  https://raw.githubusercontent.com/neuropsychology/NeuroKit/ff419d983568ef492eb8d229af643c0ef0100b32/data/eeg_resting_8min_300hz.fif \
  --output /tmp/eeg_resting_8min_300hz.fif
python examples/benchmark_channelwise.py --source /tmp/eeg_resting_8min_300hz.fif
python examples/summarize_channelwise.py
```

The full benchmark verifies the original recording's SHA-256. It uses genuinely
distinct recorded channels, not copies of a single channel. This is one recording
at order=2 and delay=4, so performance claims do not extend to all parameter sets
or EEG recordings.

For additional native-code checks:

```sh
python build_c.py --sanitize --output /tmp/sampen-ubsan.so
SAMPEN_C_LIBRARY=/tmp/sampen-ubsan.so python -m pytest tests/test_sampen_c.py -q
gcc -O1 -g -std=c11 -fopenmp -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    csrc/sampen.c tests/c_sanitizer.c -lm -o /tmp/sampen-sanitizer
/tmp/sampen-sanitizer
```


## Faster 180-second, 250 Hz, order-3 EEG

For `(64, 45000)` band-filtered EEG, use the same channel-wise API:

```python
from sampen_c import sample_entropy_c

# Example: alpha 8–13 Hz, midpoint 10.5 Hz, delay round(250/10.5)=24 samples.
entropy = sample_entropy_c(eeg_180s, order=3, delay=24, n_jobs=4)
```

Rebuild with `python build_c.py` after updating, then restart Python if the old
library was loaded. The automatic backend now chooses **exact prefix bitsets**
for order=3 with 32,768–65,536 input samples. Templates are sorted by their first
coordinate, so only a forward radius window needs checking. Neighborhoods in
the other coordinates are intersected in 64-bit words; population counts produce
exact unordered match counts. Native POPCNT is runtime-dispatched on supported
x86 CPUs, with a tested portable fallback. AVX2-capable CPUs now build masks
directly from exact ordinal ranks, reducing both runtime and prefix storage. The first-coordinate mask, partial
word boundaries, exact-radius matches, and partial prefix blocks are all handled
without approximating distances.

Select `algorithm='bitset'` explicitly for other sizes/orders, or `algorithm='tree'`
to retain the previous backend. The prefix fallback has quadratic, bit-packed storage; at 45,000 samples and
order=3 it uses roughly 14 MiB per simultaneously processed channel. The AVX2
rank path uses substantially less workspace. The automatic upper size limit bounds this
memory use. Worst-case time remains quadratic with word-level parallelism;
performance depends on the recording and radius.

The benchmark uses 180 seconds, 250 Hz, 64 distinct recorded EEG channels,
order=3 and the same four threads for both C algorithms. It filters delta 0.5–4,
theta 4–8, alpha 8–13, beta 13–30 and gamma 30–45 Hz. Delay is an integer number
of samples, computed as:

```python
delay = min(round(250 / band_midpoint_hz), int(250 * 0.125))
```

That gives **31, 31, 24, 12 and 7 samples** respectively. The largest delay is
124 ms; 32 samples would exceed the 125 ms cap. This cap belongs to the EEG
workflow; the generic library does not impose a sampling-rate-dependent limit.

[Measured timings, count validation and plot](examples/results_180s/README.md).
The recording is resampled from its original 300 Hz to match the requested
250 Hz; the benchmark documents exact filtering and cropping. Preprocessing is
excluded from the reported entropy runtime.

For reproducibility, you can inspect the **integer counts** directly:

```python
entropy, counts_m, counts_m_plus1 = sample_entropy_c(
    eeg_180s, order=3, delay=24, n_jobs=4, return_counts=True
)
```

The counts are uint64 arrays of unordered matching pairs, excluding self matches.
Use exact equality for these counts when comparing algorithms; the final
floating-point logarithm need not be bit-identical across different platforms.

To repeat the user-specific benchmark after downloading the pinned FIF:

```sh
python examples/benchmark_180s.py --source /path/to/eeg_resting_8min_300hz.fif
python examples/summarize_180s.py
```


## Latest optimization: alpha only

The latest measured update targets alpha 8–13 Hz only, preserving 180 seconds,
250 Hz, 64 channels, order=3, tolerance=0.2 × each channel SD, and a 24-sample
(96 ms) delay. It compares the previous released C library against the new one
in randomized order through the same wrapper and four threads.

[Full comparison, individual timings, exact counts and plot](examples/results_alpha/README.md).

The optimized implementation reuses one signal sort across delayed coordinates,
uses exact radix sorting for supported finite IEEE doubles, and constructs
neighborhood masks with runtime-dispatched AVX2 integer rank comparisons. The
16-bit representation is of **ordinal ranks**, not quantized EEG amplitudes;
floating-point distance/radius boundaries remain unchanged. No signal samples
or template starts are removed. Unsupported CPUs and larger template sets retain
the portable prefix-bitset implementation.

Update, run `python build_c.py`, and restart Python if the old library was loaded.
The call stays the same:

```python
entropy = sample_entropy_c(eeg, order=3, delay=24, n_jobs=4)
```

Further empirical performance comparisons use `examples/benchmark_alpha.py`;
the older five-band scripts/results are retained for historical reproducibility.
Correctness and memory-safety tests still cover other supported parameter sets.
