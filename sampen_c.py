"""Exact channel-wise C sample entropy with optional OpenMP parallelism."""
import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _library():
    path = Path(os.environ.get('SAMPEN_C_LIBRARY', Path(__file__).with_name('_sampen_c.so')))
    if not path.exists():
        raise RuntimeError('C backend is not built. Run `python build_c.py` in the repository root.')
    lib = ctypes.CDLL(str(path.resolve()))  # CDLL releases the GIL during C calls.
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.sampen_channels.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_size_t, pointer,
                                   ctypes.c_int, ctypes.c_int, pointer]
    lib.sampen_channels.restype = ctypes.c_int
    if hasattr(lib, 'sampen_channels_counts'):
        uint_pointer = ctypes.POINTER(ctypes.c_uint64)
        lib.sampen_channels_counts.argtypes = lib.sampen_channels.argtypes + [uint_pointer, uint_pointer]
        lib.sampen_channels_counts.restype = ctypes.c_int
    lib.sampen_has_openmp.argtypes = []
    lib.sampen_has_openmp.restype = ctypes.c_int
    return lib


def sample_entropy_c(data, order=2, delay=1, tolerance=None, n_jobs=1, algorithm='auto', return_counts=False):
    """Return one aligned sample entropy per channel of a (channels, time) array.

    Chebyshev distance, inclusive radius, no self matches; both template sets use
    N - order*delay starts. Zero match counts return NaN. Default tolerance is
    0.2 times each channel's population SD. An explicit tolerance may be a scalar
    shared across channels or a vector of length channels (in signal units).

    n_jobs=1 is serial; -1 uses CPUs available to the process, capped at the
    channel count. Set an explicit count when CPU quotas are smaller than CPU
    affinity. OpenMP parallelizes independent channels; it does not split one
    channel across threads. Builds without OpenMP reject n_jobs > 1.

    algorithm='auto' uses exact O(N log^2 N) range counting for order=2 and
    bit-mask counting for order=3 signals with 32768..65536 samples, and
    dual-tree counting otherwise. AVX2-capable x86 CPUs can form masks directly
    from exact 16-bit ordinal ranks, with linear workspace; other cases retain
    the portable prefix-bitset path. 'range' requires order=2. 'bitset' can be
    selected explicitly for other sizes/orders. The prefix fallback has quadratic
    bit-packed storage, so auto restricts the size range.
    return_counts=True returns (entropy, counts_m, counts_m_plus1); counts are
    exact uint64 unordered pair counts, with no tolerance-based comparison.
    First call loads the library.
    Inputs are converted to contiguous float64; no input array is modified.
    """
    for name, value, minimum in [('order', order, 2), ('delay', delay, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0 or n_jobs < -1:
        raise ValueError('n_jobs must be -1 or a positive integer')
    raw = np.asarray(data)
    if raw.ndim != 2 or np.iscomplexobj(raw):
        raise ValueError('data must be a real (channels, time) array')
    x = np.ascontiguousarray(raw, dtype=np.float64)
    channels, samples = x.shape
    if not channels or int(order)*int(delay) > samples-2 or not np.all(np.isfinite(x)):
        raise ValueError('data must have channels, finite values and at least two extendable templates')
    if tolerance is None:
        radii = .2*np.std(x, axis=1, ddof=0)
    else:
        radii = np.asarray(tolerance, dtype=np.float64)
        if radii.ndim == 0:
            radii = np.full(channels, radii.item())
        elif radii.shape != (channels,):
            raise ValueError('tolerance must be a scalar or one value per channel')
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError('tolerance must be finite and nonnegative')
    if not isinstance(return_counts, (bool, np.bool_)):
        raise ValueError('return_counts must be a boolean')
    if algorithm not in ('auto', 'tree', 'sweep', 'range', 'bitset'):
        raise ValueError("algorithm must be 'auto', 'tree', 'sweep', 'range' or 'bitset'")
    if algorithm == 'range' and order != 2:
        raise ValueError("algorithm='range' requires order=2")
    if algorithm == 'auto':
        selected = 'range' if order == 2 else 'bitset' if order == 3 and 32768 <= samples <= 65536 else 'tree'
    else:
        selected = algorithm
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    threads = min(channels, cpus if n_jobs == -1 else int(n_jobs))
    lib = _library()
    if threads > 1 and not lib.sampen_has_openmp():
        raise RuntimeError('This C library was built without OpenMP; use n_jobs=1 or rebuild with OpenMP.')
    out = np.empty(channels, dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    arguments = (x.ctypes.data_as(pointer), channels, samples, int(order), int(delay),
                 radii.ctypes.data_as(pointer), threads,
                 {'tree': 0, 'sweep': 1, 'range': 2, 'bitset': 3}[selected], out.ctypes.data_as(pointer))
    if return_counts:
        if not hasattr(lib, 'sampen_channels_counts'):
            raise RuntimeError('Rebuild the C library and restart Python to use return_counts=True.')
        counts_b = np.empty(channels, dtype=np.uint64)
        counts_a = np.empty(channels, dtype=np.uint64)
        uint_pointer = ctypes.POINTER(ctypes.c_uint64)
        status = lib.sampen_channels_counts(*arguments, counts_b.ctypes.data_as(uint_pointer),
                                           counts_a.ctypes.data_as(uint_pointer))
    else:
        status = lib.sampen_channels(*arguments)
    if status & 2:
        raise ValueError('C backend rejected the input size or values')
    if status:
        raise MemoryError('C backend could not allocate its workspace')
    return (out, counts_b, counts_a) if return_counts else out
