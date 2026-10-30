"""Summarize actual channel-wise benchmark timings without extrapolating."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(directory=ROOT/'results_channelwise'):
    rows = list(csv.DictReader((directory/'timings.csv').open()))
    metadata = json.loads((directory/'environment.json').read_text())
    groups = {}
    for row in rows:
        key = int(row['channels']), int(row['samples']), int(row['delay'])
        groups.setdefault(key, {})[row['method']] = row
    lines = ['# Channel-wise C sample entropy: empirical benchmark', '',
             'Real NeuroKit resting EEG at 300 Hz; order=2, delay=4, tolerance=0.2 × each channel SD.',
             'All numbers below are **seconds for the whole channel batch**, not per-channel times.',
             f'Median of {metadata["repeats"]} timed calls after warm-up; method order randomized.',
             f'Parallel methods use {metadata["threads"]} threads. The machine CPU quota is `{metadata["cpu_quota"]}` ',
             '(quota/period in microseconds; 400000/100000 means four CPU equivalents).',
             'Inputs are distinct recorded channels, not replicated synthetic data.', '',
             '| Channels | Samples/channel | Seconds of EEG | KDTree 1 thread | Numba 1 thread | C range 1 thread | KDTree parallel | Numba parallel | C range parallel |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    methods = ['kdtree_serial', 'numba_serial', 'c_range_serial',
               'kdtree_parallel', 'numba_parallel', 'c_range_parallel']
    for (channels, n, delay), results in groups.items():
        cells = [str(channels), f'{n:,}', f'{n/metadata["sfreq"]:.2f}']
        cells += [f'{float(results[m]["median_seconds"]):.4f}' if m in results else '—' for m in methods]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Speedups', '',
              'Ratios use the same input and include each public wrapper’s overhead.',
              'Serial C comparisons isolate algorithm/language improvements; parallel comparisons use equal thread counts.', '',
              '| Channels | Samples/channel | Serial C range vs serial Numba | Serial C range vs serial KDTree | Parallel C range vs parallel Numba | Parallel C range vs parallel KDTree |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for (channels, n, delay), results in groups.items():
        t = {m: float(r['median_seconds']) for m,r in results.items()}
        values = [f'{t["numba_serial"]/t["c_range_serial"]:.2f}×', f'{t["kdtree_serial"]/t["c_range_serial"]:.2f}×']
        values += [f'{t[m]/t["c_range_parallel"]:.2f}×' if 'c_range_parallel' in t else '—'
                   for m in ['numba_parallel', 'kdtree_parallel']]
        lines.append('| '+' | '.join([str(channels), f'{n:,}']+values)+' |')
    lines += ['', '## Serial algorithm comparison (seconds for whole batch)', '',
              '| Channels | Samples/channel | Numba sweep | C sweep | C dual tree | C range counter |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for (channels, n, delay), results in groups.items():
        cells = [str(channels), f'{n:,}']
        cells += [f'{float(results[m]["median_seconds"]):.4f}' if m in results else '—'
                  for m in ['numba_serial', 'c_sweep_serial', 'c_tree_serial', 'c_range_serial']]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'C sweep is timed only for the 8,192-sample cases; dashes denote unrun comparisons.']
    max_error = max(float(r['max_abs_entropy_error']) for r in rows)
    lines += ['', '## Correctness and limits', '',
              f'Every method was checked channel by channel against the aligned KDTree result, including every timed call.',
              f'Maximum absolute entropy difference: **{max_error:.3g}** (tolerance: rtol=atol=1e-12).',
              'Both embedding dimensions use N − order × delay starts, inclusive Chebyshev radius, and no self matches.',
              'The C backends count unordered pairs exactly; no approximate search, downsampling, or fast-math is used.', '',
              'The default C backend for order=2 uses an exact dynamic range counter with O(N log² N) runtime and O(N log N) workspace.',
              'The serial C sweep and dual-tree timings are also provided above to separate language effects from algorithm effects.',
              'OpenMP adds independent channel parallelism.',
              'Single-channel rows have no parallel measurements: channel parallelism cannot accelerate one channel.',
              'Warm-up/JIT, reading the recording and C compilation are excluded. Validation, radius computation, allocations,',
              'tree construction and the full entropy calculation are included. Parallel Numba uses a compiled `prange` loop;',
              'parallel KDTree uses a persistent Python thread pool. C starts an OpenMP team for each call.', '',
              'These are measurements on one recording, one parameter set and one shared cloud machine. They are not universal guarantees.',
              'Sweep and dual-tree methods still have quadratic worst-case work; the order-2 range method avoids that bound.',
              'More channels than available cores do not imply equal additional speedup.', '',
              '![Channel-wise timings](timings.svg)', '',
              'Bars show whole-batch medians on a logarithmic time axis; whiskers are the 25th–75th percentiles of timed calls.',
              'See [raw timings](timings.csv), [entropy vectors](entropies.json), [machine details](environment.json),',
              'and [recording provenance/license](../data/README.md).', '',
              'Reproduce (from the repository root, after downloading the pinned full FIF):', '',
              '```sh', 'python build_c.py', 'python -m pytest -q',
              'python examples/benchmark_channelwise.py --source /path/to/eeg_resting_8min_300hz.fif',
              'python examples/summarize_channelwise.py', '```', '']
    (directory/'README.md').write_text('\n'.join(line.rstrip() for line in lines))
    fig, ax = plt.subplots(figsize=(13,5.5), layout='constrained')
    chosen = ['kdtree_serial', 'numba_serial', 'c_range_serial', 'numba_parallel', 'c_range_parallel']
    labels = ['KDTree serial', 'Numba serial', 'C range serial', 'Numba parallel', 'C range parallel']
    colors = ['#999999', '#e69f00', '#56b4e9', '#cc79a7', '#0072b2']
    positions = np.arange(len(groups))
    width = .15
    for i, (method, label, color) in enumerate(zip(chosen, labels, colors)):
        for j, (key, values) in enumerate(groups.items()):
            if method not in values:
                continue
            r = values[method]
            v, low, high = [float(r[k]) for k in ['median_seconds','p25_seconds','p75_seconds']]
            ax.bar(j+(i-2)*width, v, width, color=color, label=label if j==1 else None,
                   yerr=np.array([[v-low],[high-v]]), capsize=2, error_kw={'linewidth':.8})
    ax.set_xticks(positions, [f'{ch} ch × {n:,} samples\n{n/metadata["sfreq"]:.1f} s/channel' for ch,n,d in groups])
    ax.set_yscale('log')
    ax.set_ylabel('Entire batch runtime (seconds, logarithmic; lower is faster)')
    ax.set_title(f'Exact channel-wise sample entropy on recorded resting EEG\nOrder 2 · delay 4 · parallel methods use {metadata["threads"]} threads')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis='y', alpha=.2)
    for ext in ['svg','png']:
        fig.savefig(directory/f'timings.{ext}', dpi=160)
    svg = directory/'timings.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('\n'.join(lines[:35]))


if __name__ == '__main__':
    summarize()
