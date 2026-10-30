"""Compare channel-wise C, Numba and aligned KDTree on actual resting EEG.

Default: small checked-in excerpt. --source: pinned full NeuroKit FIF, including
64-channel and long-recording cases. Warm-up excluded; all times are batch times.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampen_c import sample_entropy_c
from SampEn_utils import sample_entropy_numba, _numba_sampen_delay
from benchmark_resting_eeg import aligned_kdtree
from prepare_resting_eeg import SHA256


@njit(parallel=True, cache=True)
def _numba_parallel(x, order, delay, radii):
    result = np.empty(x.shape[0])
    for ch in prange(x.shape[0]):
        result[ch] = _numba_sampen_delay(x[ch], order, delay, radii[ch])
    return result


def numba_parallel(x, order, delay):
    # Matched bulk validation/radius computation for the C and Numba batch calls.
    x = np.ascontiguousarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('nonfinite signal')
    return _numba_parallel(x, order, delay, .2*np.std(x, axis=1))


def run(source, output, repeats=3, threads=4):
    if source is None:
        recording = np.load(ROOT/'examples/data/resting_eeg.npz', allow_pickle=False)
        data, names, fs = recording['data'], recording['channels'].tolist(), float(recording['sfreq'])
        cases = [(1, 8192, 4), (3, 8192, 4)]
        source_hash = hashlib.sha256((ROOT/'examples/data/resting_eeg.npz').read_bytes()).hexdigest()
    else:
        import mne
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_hash != SHA256:
            raise ValueError('Source recording SHA-256 mismatch')
        raw = mne.io.read_raw_fif(source, preload=False, verbose='ERROR')
        eeg = raw.copy().pick('eeg').ch_names
        names = ['C3', 'Cz', 'O1'] + [ch for ch in eeg if ch not in ['C3', 'Cz', 'O1']]
        data, fs = raw.get_data(picks=names), float(raw.info['sfreq'])
        cases = [(1, 8192, 4), (16, 8192, 4), (64, 8192, 4),
                 (8, 32768, 4), (1, 131072, 4)]
    output.mkdir(parents=True, exist_ok=True)
    numba.set_num_threads(threads)
    # Compile both serial and parallel specializations before measuring.
    warmup = np.ascontiguousarray(data[:2, :512])
    numba_parallel(warmup, 2, 4)
    sample_entropy_numba(warmup[0], delay=4)
    for algorithm in ['tree', 'sweep', 'range']:
        sample_entropy_c(warmup, delay=4, algorithm=algorithm, n_jobs=threads)
    rng = random.Random(42)
    rows, entropy_values = [], []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(lambda ch: aligned_kdtree(ch, 2, 4), warmup))
        for channels, samples, delay in cases:
            x = np.ascontiguousarray(data[:channels, :samples])
            assert x.shape == (channels, samples)
            order = 2
            functions = {
                'kdtree_serial': lambda: np.array([aligned_kdtree(ch, order, delay) for ch in x]),
                'numba_serial': lambda: np.array([sample_entropy_numba(ch, order, delay) for ch in x]),
                'c_tree_serial': lambda: sample_entropy_c(x, order, delay, algorithm='tree'),
                'c_range_serial': lambda: sample_entropy_c(x, order, delay),
            }
            if samples == 8192:
                functions['c_sweep_serial'] = lambda: sample_entropy_c(x, order, delay, algorithm='sweep')
            if channels > 1:
                functions.update({
                    'kdtree_parallel': lambda: np.array(list(pool.map(lambda ch: aligned_kdtree(ch, order, delay), x))),
                    'numba_parallel': lambda: numba_parallel(x, order, delay),
                    'c_tree_parallel': lambda: sample_entropy_c(x, order, delay, n_jobs=threads, algorithm='tree'),
                    'c_range_parallel': lambda: sample_entropy_c(x, order, delay, n_jobs=threads),
                })
            print(f'Validating {channels} channels x {samples} samples, delay={delay}', flush=True)
            expected = functions['kdtree_serial']()
            values = {}
            timings = {name: [] for name in functions}
            for repeat in range(repeats):
                methods = list(functions)
                rng.shuffle(methods)
                for name in methods:
                    start = perf_counter()
                    result = functions[name]()
                    timings[name].append(perf_counter()-start)
                    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
                    values[name] = result
                print(f'  repeat {repeat+1}/{repeats} completed', flush=True)
            for name, value in values.items():
                entropy_values.append(dict(channels=channels, samples=samples, delay=delay,
                                           method=name, values=value.tolist()))
            for name, times in timings.items():
                row = dict(channels=channels, samples=samples, duration_seconds=samples/fs,
                           order=order, delay=delay, method=name,
                           threads=threads if name.endswith('_parallel') else 1,
                           median_seconds=float(np.median(times)),
                           p25_seconds=float(np.percentile(times, 25)), p75_seconds=float(np.percentile(times, 75)),
                           max_abs_entropy_error=float(np.nanmax(np.abs(values[name]-expected))))
                rows.append(row)
                print(f'  {name}: {row["median_seconds"]:.6f} s', flush=True)
            with (output/'timings.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            (output/'entropies.json').write_text(json.dumps(entropy_values, indent=2)+'\n')
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    quota = Path('/sys/fs/cgroup/cpu.max')
    metadata = dict(python=sys.version, platform=platform.platform(), cpu=cpu,
                    cpu_quota=quota.read_text().strip() if quota.exists() else 'unknown',
                    threads=threads, repeats=repeats, sfreq=fs, channel_order=names,
                    source='NeuroKit full resting FIF' if source else 'checked-in 30-second excerpt',
                    source_sha256=source_hash, preprocessing='First N samples; no added filtering, re-referencing or resampling.',
                    versions={p: importlib.metadata.version(p) for p in ['numpy','numba','llvmlite','scikit-learn','mne']},
                    numba_threading_layer=numba.threading_layer(),
                    timing='Entire channel batch; wrappers, validation, radius calculation, allocations and tree builds included; '
                           'warm-up/JIT excluded. Randomized method order, seed=42. Parallel KDTree uses a persistent thread pool; '
                           'parallel Numba uses a compiled prange channel loop; C uses OpenMP.',
                    compiler='gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp; no fast-math or march=native',
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Full pinned NeuroKit resting FIF; enables long/many-channel cases')
    parser.add_argument('--output', type=Path, default=ROOT/'examples/results_channelwise')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive')
    run(args.source, args.output, args.repeats, args.threads)
