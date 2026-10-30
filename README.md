# Sample entropy with delay

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
