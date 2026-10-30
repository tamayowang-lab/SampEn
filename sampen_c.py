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
    lib.sampen_has_openmp.argtypes = []
    lib.sampen_has_openmp.restype = ctypes.c_int
    return lib


def sample_entropy_c(data, order=2, delay=1, tolerance=None, n_jobs=1, algorithm='auto'):
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
    dual-tree counting for higher orders. 'range' requires order=2; 'tree' and
    'sweep' provide controlled comparisons and have quadratic worst-case time.
    The range method uses O(N log N) workspace. First call loads the library.
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
    if algorithm not in ('auto', 'tree', 'sweep', 'range'):
        raise ValueError("algorithm must be 'auto', 'tree', 'sweep' or 'range'")
    if algorithm == 'range' and order != 2:
        raise ValueError("algorithm='range' requires order=2")
    selected = 'range' if algorithm == 'auto' and order == 2 else 'tree' if algorithm == 'auto' else algorithm
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    threads = min(channels, cpus if n_jobs == -1 else int(n_jobs))
    lib = _library()
    if threads > 1 and not lib.sampen_has_openmp():
        raise RuntimeError('This C library was built without OpenMP; use n_jobs=1 or rebuild with OpenMP.')
    out = np.empty(channels, dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = lib.sampen_channels(x.ctypes.data_as(pointer), channels, samples, int(order), int(delay),
                                 radii.ctypes.data_as(pointer), threads, {'tree': 0, 'sweep': 1, 'range': 2}[selected],
                                 out.ctypes.data_as(pointer))
    if status & 2:
        raise ValueError('C backend rejected the input size or values')
    if status:
        raise MemoryError('C backend could not allocate its workspace')
    return out
