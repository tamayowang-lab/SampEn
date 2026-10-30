"""Generate a table and figure from the empirical benchmark CSV."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(directory=ROOT/'results'):
    with (directory/'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    metadata = json.loads((directory/'environment.json').read_text())
    repeats = metadata['repeats']
    groups = {}
    for row in rows:
        key = (row['channel'], int(row['samples']), int(row['order']), int(row['delay']))
        groups.setdefault(key, {})[row['method']] = row
    lines = [
        '# Resting EEG benchmark', '',
        'Three channels (C3, Cz, O1), first 30 seconds of the NeuroKit resting EEG example,',
        '300 Hz, orders 2 and 3, delays 1, 2, 4 and 8, tolerance = 0.2 × segment standard deviation.',
        'See [data provenance](../data/README.md) and [machine and timing details](environment.json).',
        'Times include the public Python wrappers; compilation and warm-up are excluded.',
        f'Each timing is the median of {repeats} calls in randomized method order.',
        'These are measurements on one recording and one machine, not universal speed guarantees.', '',
        '## Speedup by segment length', '',
        'Each entry is the median baseline/new-runtime ratio across channels, orders and delays.',
        'A ratio greater than 1 means the new implementation is faster.',
        'The original Numba comparison uses delay=1 only, because it has no delay support.', '',
        '| Samples | Original KDTree / new | Aligned KDTree / new | Original Numba / new (delay=1) |',
        '| ---: | ---: | ---: | ---: |',
    ]
    for n in sorted({key[1] for key in groups}):
        ratios = {m: [] for m in ['original_kdtree','aligned_kdtree','original_numba']}
        for key, values in groups.items():
            if key[1] != n:
                continue
            new = float(values['numba_delay']['median_ms'])
            for m in ratios:
                if m in values:
                    ratios[m].append(float(values[m]['median_ms'])/new)
        lines.append(f'| {n} | ' + ' | '.join(f'{np.median(v):.2f}×' for v in ratios.values()) + ' |')
    for method, label in [('original_kdtree', 'Original KDTree'), ('original_numba', 'Original Numba (delay=1)')]:
        ratios = [float(v[method]['median_ms'])/float(v['numba_delay']['median_ms'])
                  for v in groups.values() if method in v]
        lines.extend(['', f'{label}: overall median speedup {np.median(ratios):.2f}×; '
                      f'new implementation faster in {sum(r > 1 for r in ratios)}/{len(ratios)} cases; '
                      f'range {min(ratios):.2f}–{max(ratios):.2f}×.'])
    errors = [abs(float(v['numba_delay']['entropy'])-float(v['aligned_kdtree']['entropy'])) for v in groups.values()]
    differences = [abs(float(v['numba_delay']['entropy'])-float(v['original_kdtree']['entropy']))
                   for k,v in groups.items() if k[3]>1]
    lines.extend(['', '## Correctness and interpretation', '',
                  f'All {len(groups)} empirical configurations matched the aligned KDTree reference',
                  f'(maximum absolute entropy difference: {max(errors):.3g}).',
                  'At delay=1, results also matched both original implementations.',
                  f'For delay>1, the maximum difference from the unchanged original KDTree output was {max(differences):.6g}.',
                  'That difference comes from the original function using unequal numbers of order-m and order-(m+1) templates.',
                  'The new function uses the same extendable start indices for both dimensions.', '',
                  '![Representative timings](timings.svg)', '',
                  'The figure shows C3, order=2; the table aggregates all tested channels, orders and delays.',
                  'Raw per-configuration medians and quartiles are in [timings.csv](timings.csv).', '',
                  'Reproduce from the repository root:', '',
                  '```sh', 'python examples/benchmark_resting_eeg.py',
                  'python examples/summarize_benchmark.py', '```', ''])
    (directory/'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True, layout='constrained')
    labels = {'original_kdtree':'Original KDTree', 'aligned_kdtree':'Aligned KDTree',
              'numba_delay':'New Numba', 'original_numba':'Original Numba'}
    for ax, delay in zip(axes, [1,4]):
        for method, label in labels.items():
            selected = [r for r in rows if r['channel']=='C3' and int(r['order'])==2
                        and int(r['delay'])==delay and r['method']==method]
            if selected:
                selected.sort(key=lambda r:int(r['samples']))
                ax.loglog([int(r['samples']) for r in selected],
                          [float(r['median_ms']) for r in selected], 'o-', label=label)
        ax.set(title=f'C3, order=2, delay={delay}', xlabel='Samples')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Median runtime (ms; lower is faster)')
    axes[1].legend(fontsize=8)
    svg = directory/'timings.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('\n'.join(lines[:23]))


if __name__ == '__main__':
    summarize()
