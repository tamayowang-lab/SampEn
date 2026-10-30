"""Benchmark the checked-in empirical EEG example; run from any directory."""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SampEn_utils import sample_entropy, sample_entropy_numba, _numba_sampen


def aligned_kdtree(x, order=2, delay=1, tolerance=None):
    """Same KDTree algorithm as original, with corrected template alignment."""
    r = .2*np.std(x) if tolerance is None else tolerance
    n = len(x) - order*delay
    counts = []
    for m in [order, order+1]:
        templates = x[np.arange(n)[:, None] + np.arange(m)*delay]
        counts.append(np.sum(KDTree(templates, metric='chebyshev').query_radius(
            templates, r, count_only=True)-1))
    return -np.log(counts[1]/counts[0]) if all(counts) else np.nan


def benchmark(data_path, output, repeats=7):
    dataset = np.load(data_path, allow_pickle=False)
    data, channels = dataset['data'], dataset['channels']
    fs = float(dataset['sfreq'])
    sample = np.ascontiguousarray(data[0, :512])
    start = perf_counter()
    sample_entropy_numba(sample)
    first_call = perf_counter()-start
    rows = []
    randomizer = random.Random(42)
    for channel, signal in zip(channels, data):
        for n in [256, 512, 1024, 2048, 4096, 8192]:
            if n > len(signal):
                continue
            x = np.ascontiguousarray(signal[:n], dtype=np.float64)
            for order in [2, 3]:
                for delay in [1, 2, 4, 8]:
                    r = .2*np.std(x)
                    methods = {
                        'original_kdtree': lambda: sample_entropy(x, order, delay, r),
                        'aligned_kdtree': lambda: aligned_kdtree(x, order, delay, r),
                        'numba_delay': lambda: sample_entropy_numba(x, order, delay, r),
                    }
                    if delay == 1:
                        methods['original_numba'] = lambda: _numba_sampen(x, order, r)
                    values = {name: fn() for name, fn in methods.items()}
                    np.testing.assert_allclose(values['numba_delay'], values['aligned_kdtree'],
                                               rtol=1e-12, atol=1e-12, equal_nan=True)
                    if delay == 1:
                        for name in ['original_kdtree', 'original_numba']:
                            np.testing.assert_allclose(values['numba_delay'], values[name],
                                                       rtol=1e-12, atol=1e-12, equal_nan=True)
                    timings = {name: [] for name in methods}
                    for _ in range(repeats):
                        names = list(methods)
                        randomizer.shuffle(names)
                        for name in names:
                            start = perf_counter()
                            methods[name]()
                            timings[name].append((perf_counter()-start)*1000)
                    for name, times in timings.items():
                        rows.append(dict(channel=str(channel), samples=n, seconds=n/fs,
                                         order=order, delay=delay, tolerance=r, method=name,
                                         entropy=float(values[name]), median_ms=float(np.median(times)),
                                         p25_ms=float(np.percentile(times, 25)),
                                         p75_ms=float(np.percentile(times, 75))))
        print(f'Finished {channel}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    metadata = dict(python=sys.version, platform=platform.platform(), cpu=cpu,
                    versions={p: importlib.metadata.version(p) for p in ['numpy','scipy','numba','llvmlite','scikit-learn']},
                    repeats=repeats, first_numba_call_seconds=first_call,
                    first_call_note='May load disk cache; not a clean-cache compilation measurement.',
                    sfreq=fs, data_path=str(data_path),
                    data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    completed_at_utc=datetime.now(timezone.utc).isoformat(), random_seed=42,
                    preprocessing='None: native sample rate, signal in volts, first N samples.',
                    timing='Single process, single-threaded kernels; randomized method order; warm-up excluded; public wrappers included.')
    (output/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'examples/data/resting_eeg.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'examples/results')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    benchmark(args.data, args.output, args.repeats)
