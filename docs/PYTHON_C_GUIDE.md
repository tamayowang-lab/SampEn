# Complete guide: calling the C sample-entropy implementation from Python

This guide uses your workflow: **250 Hz, 180 seconds, 64 channels, order 3,
alpha 8–13 Hz, delay 24 samples (96 ms)**. The delay is below your 125 ms cap.
The input for one segment has shape **(64, 45000)**, and the output has shape
**(64,)**. All C methods use the corrected alignment.

[Fresh first-versus-latest benchmark and plot](../examples/results_first_vs_latest/README.md)
compares the aligned KDTree reference, the first corrected Numba kernel, the
first C release and the latest C implementation on the identical EEG array.
Only alpha is used for these empirical timing tests.

## 1. What Python actually runs

There are three parts:

| File | Purpose |
| --- | --- |
| `csrc/sampen.c` and its headers | Native C algorithms, including exact range/mask counters and channel parallelism. |
| `_sampen_c.so` | Compiled shared library produced locally by `build_c.py`. |
| `sampen_c.py` | Python/NumPy wrapper that calls the library through `ctypes`. |

You compile the C source once, then import `sample_entropy_c` in Python. Python
passes the whole channels-by-time array to C in one call. The C function handles
the channel loop; OpenMP can compute independent channels simultaneously. The
`ctypes.CDLL` call releases the Python GIL during native computation.

You do not run the C source as a separate command for every channel. The shared
library is a local build output and is not committed to Git; rebuild it on each
machine and after updating C code. Restart Python/Jupyter after rebuilding a
library that was already loaded.

## 2. Get the project and build it

The tested environment is Linux x86-64, Python 3.12 and GCC with OpenMP.
You need Git, Python with `venv`, and a C compiler with its OpenMP runtime.
On Debian/Ubuntu, `build-essential` supplies GCC; `libgomp1` is its OpenMP runtime.
The appropriate Python/venv packages depend on your distribution. The commands
below assume `python3.12` is already available.

For a fresh copy:

```sh
git clone --branch numba-delay-eeg-benchmark https://github.com/tamayowang-lab/SampEn.git
cd SampEn
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python build_c.py
python -m pytest -q
```

`requirements-dev.txt` installs the dependencies for the complete guide, tests
and EEG benchmarks. To use only the C wrapper, the Python runtime dependency is
NumPy; the C library does not need NumPy C headers or Python development headers:

```sh
python -m pip install numpy==2.3.5
python build_c.py
```

Successful compilation creates `_sampen_c.so` beside `sampen_c.py`. The build uses
`-O3`, `-std=c11`, `-fPIC`, `-shared`, `-fopenmp` and `-lm`. It does not enable
fast-math or require a global `-march=native` setting. AVX2/POPCNT instructions
are used only in runtime-dispatched functions on CPUs that support them.

To update an existing checkout that is already on the project branch:

```sh
git fetch origin
git switch numba-delay-eeg-benchmark
git pull --ff-only
source .venv/bin/activate
python build_c.py
```

Then restart your Python process or Jupyter kernel. Git will stop if local work
prevents switching/updating; preserve that work before resolving the conflict.

The published commands are Linux-oriented. Windows users can use WSL2 and run
both Python and GCC inside that Linux environment. Native Windows builds have
not been validated. For environments without OpenMP, a serial build is available:

```sh
CC=clang python build_c.py --no-openmp
```

Use `n_jobs=1` with a serial build. Platform/compiler availability still needs
to be checked on your machine; the measured results in this repository are Linux/GCC results.

## 3. Quick offline smoke check

The repository includes a small real EEG excerpt. It is **3 channels × 9000
samples at 300 Hz**, rather than the full 64-channel alpha benchmark. It is
useful for checking that compilation and import work without a download:

```python
import numpy as np
from sampen_c import sample_entropy_c

example = np.load("examples/data/resting_eeg.npz", allow_pickle=False)
x = np.ascontiguousarray(example["data"], dtype=np.float64)
values = sample_entropy_c(x, order=3, delay=1, n_jobs=1)

print(x.shape)       # (3, 9000)
print(values.shape)  # (3,)
print(values)
```

These are API smoke-check settings, not the alpha timing protocol.

## 4. Calculate entropy for your own alpha EEG

This example assumes the file already contains **alpha-filtered EEG at 250 Hz**.
The C function itself performs no filtering, resampling, artifact correction or
re-referencing.

```python
import numpy as np
from sampen_c import sample_entropy_c

fs = 250
seconds = 180
n_samples = fs * seconds

# Rows are channels, columns are time samples.
eeg = np.load("your_alpha_250hz.npy", allow_pickle=False)
if eeg.ndim != 2:
    raise ValueError("Expected a 2D channels-by-time array")
if eeg.shape[1] < n_samples:
    raise ValueError("Need at least 45,000 time samples per channel")

# Example: take the first 180 seconds. For another epoch, change the slice.
eeg = np.ascontiguousarray(eeg[:, :n_samples], dtype=np.float64)
if not np.all(np.isfinite(eeg)):
    raise ValueError("Handle NaN/Inf values before computing entropy")

band_midpoint = (8 + 13) / 2
max_delay_samples = int(np.floor(fs * 0.125))  # 31 at 250 Hz
# Delay is an INTEGER NUMBER OF SAMPLES, not seconds.
delay = min(round(fs / band_midpoint), max_delay_samples)  # 24

sampen = sample_entropy_c(
    eeg,
    order=3,
    delay=delay,
    tolerance=None,
    n_jobs=4,
    algorithm="auto",
)

assert sampen.shape == (eeg.shape[0],)
print(sampen)
```

For your usual input, `eeg.shape == (64, 45000)`, the output contains 64 values
in the same channel order. A `(45000, 64)` time-by-channel array must be transposed
before calling. For one channel, use `signal[None, :]` to retain the 2D format.

Calling once for all channels lets C parallelize them. A Python loop that calls
C separately for each single channel cannot use this channel parallelism.

The wrapper converts inputs to contiguous float64 and does not modify them.
Preparing that layout once is useful when you will make repeated calls.

### Importing from a different project directory

This repository currently provides an importable module rather than an
installable Python package. Run your script from the checkout, or add its
absolute path to Python's import path:

```python
import sys
sys.path.insert(0, "/absolute/path/to/SampEn")
from sampen_c import sample_entropy_c
```

The wrapper finds `_sampen_c.so` relative to `sampen_c.py`, not relative to your
script's working directory.

## 5. Parameters, units and output conventions

| Parameter | Meaning |
| --- | --- |
| `data` | Real, finite 2D array, shape `(channels, time)`. |
| `order` | Embedding dimension, integer at least 2. Your setting is 3. |
| `delay` | Positive integer spacing between coordinates, in samples. Your alpha setting is 24. |
| `tolerance=None` | Uses `0.2 * np.std(channel, ddof=0)` independently for each channel. |
| `tolerance=scalar` | Uses the same absolute radius for every channel. |
| `tolerance=vector` | One absolute radius per channel, shape `(channels,)`. |
| `n_jobs` | Channel workers: 1 is serial; a positive number requests that many workers, capped at channel count. |
| `algorithm="auto"` | Chooses the implemented backend for the input shape/order. Recommended for your workflow. |
| `return_counts=True` | Also returns exact uint64 unordered pair counts for dimensions `m` and `m+1`. |

The C implementation uses **Chebyshev distance**, includes distances equal to
the radius, and excludes self matches. Radius values use the same units as your
data. For example, MNE usually returns EEG in volts; an explicit tolerance must
then also be in volts. With the default, the radius is derived from each channel.

For order `m`, delay `d` and length `N`, both template sets use the same
`N - m*d` starting positions. For your case, that is `45000 - 3*24 = 44928`.
The shorter template is `[x[t], x[t+24], x[t+48]]`; the extension adds `x[t+72]`.
All valid starting positions are used; the function does not downsample by the delay.

If `B` is the number of matching shorter-template pairs and `A` is the number of
matching extended-template pairs, the entropy is `-log(A/B)`, in natural-log units.
Both counts are unordered and exclude self matches. Their common normalization
factor cancels in the ratio.

A constant finite channel returns 0. A zero `A` or `B` returns NaN, following this
project's convention; some other packages choose infinity for particular cases.
The function rejects nonfinite data/radii, negative radii, invalid integer
parameters, and fewer than two extendable templates. It does not replace missing data.

### Choosing a worker count

The cloud benchmark has a four-CPU-equivalent quota, so it uses `n_jobs=4`.
On your machine choose a worker count appropriate to the CPUs available to your
process. `n_jobs=-1` uses CPU affinity, which may be larger than a container's
CPU quota. More workers than available CPU capacity can reduce performance.
A single channel uses one worker even if you request more.

The first library load and OpenMP initialization have startup cost. No JIT is
needed for the C backend; Numba baseline compilation is excluded from warm timings.

### Multiple epochs

For arrays shaped `(epochs, channels, time)`, call once per epoch:

```python
values = np.stack([
    sample_entropy_c(epoch, order=3, delay=24, n_jobs=4)
    for epoch in alpha_epochs
])
# values.shape == (epochs, channels)
```

If you parallelize epochs using separate processes, allocate inner worker counts
accordingly so the combined workers fit your CPU budget.

## 6. Optional alpha filtering from broadband data

If your data is already alpha-filtered, use it directly. To follow the benchmark's
filter design on broadband data that is already sampled at 250 Hz:

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt

# broadband has shape (channels, time), sampled at 250 Hz.
sos = butter(4, [8, 13], btype="bandpass", fs=250, output="sos")
alpha_full = sosfiltfilt(sos, broadband, axis=1)
alpha_epoch = np.ascontiguousarray(alpha_full[:, :45000], dtype=np.float64)
```

The benchmark resamples the original public recording from 300 to 250 Hz using
`resample_poly(up=5, down=6)`, filters the complete eight-minute recording, then
extracts the first 180 seconds. Filtering an isolated epoch can produce different
edge values. Use the preprocessing required by your analysis; comparisons need
the identical preprocessed array for all algorithms.

## 7. Time the latest C implementation on your own data

Use wall-clock elapsed time because the native calculation uses multiple threads.
This measures entropy computation after your input array has been prepared:

```python
from time import perf_counter
import numpy as np
from sampen_c import sample_entropy_c

# eeg is the prepared (channels, time) alpha array.
kwargs = dict(order=3, delay=24, n_jobs=4)
sample_entropy_c(eeg, **kwargs)  # warm library and thread initialization

seconds = []
for _ in range(11):
    start = perf_counter()
    values = sample_entropy_c(eeg, **kwargs)
    seconds.append(perf_counter() - start)

print("Input shape:", eeg.shape)
print("Median batch seconds:", np.median(seconds))
print("25th–75th percentiles:", np.percentile(seconds, [25, 75]))
```

Keep printing, file reading, filtering, resampling and compilation outside the
timer when comparing algorithms. The call still includes validation, tolerance
calculation, allocations and the complete native entropy calculation. Report
whether you measure cold startup or warmed calls, and keep thread counts equal.

The saved benchmark uses randomized method order and warm-up for every method,
then compares medians and quartiles. The recorded data is empirical EEG, not
synthetic noise or tiled copies of a channel. Only alpha is timed.

## 8. Validate results and inspect the integer counts

```python
import numpy as np
from sampen_c import sample_entropy_c

latest, b_latest, a_latest = sample_entropy_c(
    eeg, order=3, delay=24, n_jobs=4, return_counts=True
)
reference, b_reference, a_reference = sample_entropy_c(
    eeg, order=3, delay=24, n_jobs=4,
    algorithm="tree", return_counts=True
)

np.testing.assert_array_equal(b_latest, b_reference)
np.testing.assert_array_equal(a_latest, a_reference)
np.testing.assert_allclose(latest, reference, rtol=1e-12, atol=1e-12, equal_nan=True)
```

Counts use exact equality. Floating-point logarithms can vary slightly between
libraries/platforms, so a small tolerance is appropriate for entropy comparisons.
A nonzero count difference needs investigation; it is not a logarithm-rounding effect.

The benchmark also computes independent aligned KDTree counts for **every channel**
and compares them with the latest C counts. The historical first C/Numba APIs
return entropy only, so the report does not claim direct access to their raw counts.

The legacy `sample_entropy()` function is preserved with its original behavior,
including the delay>1 alignment issue discussed in the README. The corrected
Numba entry point is `sample_entropy_numba()`. The comparison script uses aligned
KDTree templates, not that uncorrected legacy output.

## 9. Reproduce first-correct-versus-latest timing

“First correct” can mean different milestones. The benchmark includes both:

| Method | Definition |
| --- | --- |
| Aligned KDTree | Independent corrected reference, with the same starts in both dimensions. |
| First corrected Numba | Unmodified delay-aware kernel from commit `76fa55f`; a compiled channel adapter supplies four-thread parallelism. |
| First C | Unmodified compiled source from commit `4b55923`, using its correct channel-wise dual-tree implementation. |
| Latest C | Current checkout's compiled backend, `algorithm="auto"`. |

The Numba adapter preserves the original counting kernel and only adds a parallel
channel loop plus batch validation/radius calculation. The first C library is
loaded separately through the same Python wrapper with its original supported
`algorithm="tree"`. The old library does not expose raw count output. None of
these steps resets your active checkout or changes the historical source.

Install full test/benchmark dependencies and build the latest library:

```sh
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python build_c.py
```

Download the pinned public recording (approximately 70 MiB):

```sh
mkdir -p /tmp/sampen-guide-data
curl --fail --location \
  https://raw.githubusercontent.com/neuropsychology/NeuroKit/ff419d983568ef492eb8d229af643c0ef0100b32/data/eeg_resting_8min_300hz.fif \
  --output /tmp/sampen-guide-data/eeg_resting_8min_300hz.fif
```

The benchmark verifies the source SHA-256 before reading it. See
[recording attribution and license](../examples/data/README.md).

Extract historical source into a temporary directory and compile the first C:

```sh
mkdir -p /tmp/sampen-first-c /tmp/sampen-first-numba
git archive --format=tar --output=/tmp/sampen-first-c/source.tar 4b55923 csrc
tar -xf /tmp/sampen-first-c/source.tar -C /tmp/sampen-first-c
gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp \
    /tmp/sampen-first-c/csrc/sampen.c -lm -o /tmp/sampen-first-c/first_c.so
git show 76fa55f:SampEn_utils.py > /tmp/sampen-first-numba/SampEn_utils.py
```

Run the full comparison:

```sh
python examples/benchmark_first_vs_latest.py \
    --source /tmp/sampen-guide-data/eeg_resting_8min_300hz.fif \
    --first-c-lib /tmp/sampen-first-c/first_c.so \
    --first-numba-file /tmp/sampen-first-numba/SampEn_utils.py \
    --threads 4 --repeats 5
python examples/summarize_first_vs_latest.py
```

The first NumPy/Numba import and JIT compilation happen before timing. Each method
also receives a full-batch warm-up. The older methods are much slower, so this
comparison takes several minutes. Every timed entropy vector is checked against
the independent aligned reference; failed checks stop the run.

Results are saved under `examples/results_first_vs_latest/`: medians, quartiles,
every individual timing, validation counts, environment/source fingerprints and
an exportable plot. The benchmark does not claim to measure filtering or full
application startup. The timing values come from one public recording and one
machine; your data and hardware can differ.

## 10. Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError: sampen_c` | Run from the checkout or add its absolute path to `sys.path`; this is not currently installed by `pip install .`. |
| “C backend is not built” | Run `python build_c.py` in the checkout. |
| GCC/OpenMP compilation fails | Check the compiler error and install the compiler/runtime; `CC=...` selects a compiler. A `--no-openmp` build supports serial use. |
| `libgomp.so.1` is missing | Install the GCC OpenMP runtime, e.g. `libgomp1` on Debian/Ubuntu. |
| Parallel request rejected | Rebuild with OpenMP, or call with `n_jobs=1`. |
| Results/timing still look like an older version | Restart Python/Jupyter after rebuilding; an already-loaded shared library remains mapped. |
| Input shape is wrong | Arrange data as channels × time; do not pass a 3D epoch array directly. |
| Some entropy outputs are NaN | Check the exact counts and data properties; do not silently replace them with zero. |
| Slower than the saved benchmark | Check input length/order, CPU features, worker count, CPU quota and other running work. Auto selection and hardware differ. |
| Historical `git archive`/`git show` ref is unavailable | Use a checkout containing history for the listed commits; a shallow clone may need additional history. |

The optimized order-3 automatic path targets 32,768–65,536 input samples. Your
45,000-sample epochs are inside this range. Other sizes/orders remain supported
but can select a different backend. AVX2/POPCNT acceleration is automatic on
supported CPUs, with portable fallbacks; no unsafe CPU instruction flags are
required from the caller.
