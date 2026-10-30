"""Create the first-correct versus latest alpha EEG report and chart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=[('aligned_kdtree','Aligned KDTree reference'),
         ('first_correct_numba_76fa55f','First corrected Numba kernel (76fa55f)'),
         ('first_correct_c_4b55923','First C (4b55923)'),('latest_c','Latest C')]


def summarize(directory=ROOT/'results_first_vs_latest'):
    timing=json.loads((directory/'timings.json').read_text())
    metadata=json.loads((directory/'environment.json').read_text())
    summary=timing['summary'];latest=summary['latest_c']['median_seconds']
    lines=['# First correct implementations versus latest C','',
           '**One identical empirical alpha EEG array:** 64 channels × 45,000 samples, 180 seconds at 250 Hz.',
           'Band 8–13 Hz, order=3, delay=24 samples (96 ms, below the 125 ms cap), tolerance=0.2 × each channel population SD.',
           f'Every method uses the same **{metadata["threads"]}-thread budget**.',
           f'Medians and quartiles come from {metadata["repeats"]} timed repetitions, in randomized method order, after full-batch warm-up.',
           'All times are **seconds for the entire 64-channel batch**, excluding reading/filtering/resampling and JIT/startup.','',
           '| Implementation | Median (s) | 25th–75th percentile (s) | Latest speedup |',
           '| --- | ---: | --- | ---: |']
    for key,label in METHODS:
        r=summary[key]
        lines.append(f'| {label} | {r["median_seconds"]:.4f} | {r["p25_seconds"]:.4f}–{r["p75_seconds"]:.4f} | {r["median_seconds"]/latest:.2f}× |')
    max_error=max(r['max_abs_entropy_error'] for r in summary.values())
    lines+=['','## What “first correct” means','',
            'The earliest delay-aware corrected **Numba** implementation is archived from commit `76fa55f`.',
            'Its unchanged counting kernel is wrapped in a compiled `prange` channel loop, giving it the same worker budget as C.',
            'This adapter only adds batch validation, tolerance calculation and independent channel parallelism.',
            'The original 1D API did not accept a channels-by-time array; a serial Python loop would be a different comparison.','',
            'The first correctly aligned channel-wise **C** implementation is compiled unchanged from commit `4b55923`.',
            'It is loaded separately and called with its original supported `algorithm="tree"` through the same Python wrapper.',
            'The latest C kernel is from native-code commit `5982e46` in this measured run, using `algorithm="auto"`.',
            'Both C libraries use GCC -O3 with OpenMP, without fast-math or a global march=native option.','',
            'The aligned KDTree reference also uses the corrected template starts. This comparison does not use the',
            'uncorrected legacy `sample_entropy()` output at delay>1.','',
            '## Numerical validation','',
            f'Every timed entropy vector agreed with the aligned KDTree reference (maximum observed absolute difference: **{max_error:.3g}**).',
            'Checks used rtol=atol=1e-12 and matching NaN positions; zero observed difference in this run is not a universal',
            'cross-platform bitwise-equality guarantee.','',
            'The latest C uint64 counts for **both embedding dimensions** exactly matched independent KDTree counts for all 64 channels.',
            'The historical first Numba/C APIs only return entropy, so direct raw-count access for those old binaries is not claimed.',
            '[Counts and entropy](counts.json) · [All individual timings](timings.json) · [Environment and code fingerprints](environment.json)','',
            '## What is included','',
            'The measurements include wrapper work, tolerance calculation, allocations, template/tree construction where applicable,',
            'and the complete entropy calculation. They time the entropy-only public C call, matching ordinary use.',
            'The latest raw counts are obtained separately for validation. Full-batch warm-up is performed for every method.',
            'Imports, Numba compilation, file reading, resampling and band filtering are excluded.','',
            'The public NeuroKit recording is resampled from 300 to 250 Hz using resample_poly(5,6), then filtered using',
            'butter(N=4, bandpass=[8,13], output="sos") and sosfiltfilt on the complete eight-minute recording.',
            'The first 180 seconds are selected afterward. Inputs are distinct recorded channels, not replicated signals.',
            'See [recording provenance and license](../data/README.md). Results can vary with data and hardware.','',
            '![First correct versus latest implementations](timings.svg)','',
            'Left: all implementations on a logarithmic time axis. Right: first C versus latest C on a linear axis.',
            'Bars show batch medians; whiskers show the 25th–75th percentiles.',
            '[PNG](timings.png) · [SVG](timings.svg)','',
            '## Build, use and reproduce','',
            'The [complete Python/C guide](../../docs/PYTHON_C_GUIDE.md) contains installation, data preparation, parameter definitions,',
            'own-data timing, exact-count checks, troubleshooting and the historical-source compilation commands.',
            'The benchmark entry point is `examples/benchmark_first_vs_latest.py`; the plot/report generator is',
            '`examples/summarize_first_vs_latest.py`. Neither resets the active checkout.','']
    (directory/'README.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    values=np.array([summary[key]['median_seconds'] for key,label in METHODS])
    lo=np.array([summary[key]['p25_seconds'] for key,label in METHODS]);hi=np.array([summary[key]['p75_seconds'] for key,label in METHODS])
    bars=axes[0].bar(np.arange(4),values,color=['#999999','#e69f00','#56b4e9','#0072b2'],
                     yerr=[values-lo,hi-values],capsize=4)
    axes[0].bar_label(bars,labels=[f'{v:.3f} s' for v in values],padding=6,fontsize=9)
    axes[0].set_yscale('log');axes[0].set_ylim(min(values)*.65,max(values)*1.65)
    axes[0].set_xticks(range(4),['Aligned\nKDTree','First corrected\nNumba','First C\n4b55923','Latest C\n5982e46'],fontsize=9)
    axes[0].set_ylabel('Whole-batch seconds (logarithmic; lower is faster)')
    axes[0].set_title('All correctly aligned implementations')
    ix=np.array([2,3]);v=values[ix]
    bars=axes[1].bar([0,1],v,color=['#56b4e9','#0072b2'],width=.55,
                     yerr=[v-lo[ix],hi[ix]-v],capsize=4)
    axes[1].bar_label(bars,labels=[f'{a:.3f} s' for a in v],padding=6,fontsize=11)
    axes[1].set_xticks([0,1],['First C\n4b55923','Latest C\n5982e46'])
    axes[1].set_ylim(0,max(v)*1.22);axes[1].set_ylabel('Whole-batch seconds (linear)')
    axes[1].set_title(f'Latest versus first C: {v[0]/v[1]:.1f}× faster')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Alpha EEG · 64 channels · 180 s at 250 Hz · order 3 · delay 24',fontsize=13)
    fig.supxlabel(f'Matched {metadata["threads"]} threads · {metadata["repeats"]} timed trials per method · preprocessing/JIT excluded',fontsize=10)
    for ext in ['png','svg']:fig.savefig(directory/f'timings.{ext}',dpi=170)
    svg=directory/'timings.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('\n'.join(lines[:16]))


if __name__=='__main__':summarize()
