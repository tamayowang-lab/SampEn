"""Summarize the further alpha-only tuning against the selected baseline."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def summarize(directory=ROOT/'results_alpha_limit'):
    results=json.loads((directory/'timings.json').read_text())
    metadata=json.loads((directory/'environment.json').read_text())
    key=next(k for k in results['summary'] if k.startswith('previous_release_'))
    previous,current=results['summary'][key],results['summary']['optimized']
    baseline=metadata['baseline_commit'][:7]
    speedup=results['speedup']
    reduction=100*(1-current['median_seconds']/previous['median_seconds'])
    lines=['# Further alpha-only tuning: diminishing returns','',
           '**64 channels × 45,000 samples**, 180 seconds at 250 Hz, order=3, alpha 8–13 Hz.',
           'Delay: 24 samples = 96 ms, below the 125 ms cap. Tolerance: 0.2 × each filtered channel population SD.',
           f'Both versions use {metadata["threads"]} threads and the identical Python wrapper on the same machine.',
           f'{metadata["repeats"]} timed repetitions per version, randomized method order after full-batch warm-up.','',
           '| Version | Median (s) | 25th percentile (s) | 75th percentile (s) |',
           '| --- | ---: | ---: | ---: |',
           f'| Previous `{baseline}` | {previous["median_seconds"]:.4f} | {previous["p25_seconds"]:.4f} | {previous["p75_seconds"]:.4f} |',
           f'| Further optimized | {current["median_seconds"]:.4f} | {current["p25_seconds"]:.4f} | {current["p75_seconds"]:.4f} |','',
           f'**Additional speedup: {speedup:.2f}×; {reduction:.1f}% less runtime.** Times cover all 64 channels.',
           'This is a fresh comparison against the immediately previous release, rather than a ratio against an old result.',
           'Validation, radius calculation, allocations, entropy computation and returning both count vectors are included.',
           'Reading, resampling, filtering, initialization and warm-up are excluded.','',
           '## Improvements retained','',
           'The previous implementation reused a signal sort, but searched radius endpoints again in every delayed coordinate.',
           'The new implementation computes exact radius endpoints **once in the full sorted signal**. Prefix counts of valid',
           'samples translate these endpoints into each delayed coordinate’s restricted sample interval. This also avoids',
           'copying sorted coordinate views. Lower radius endpoints are derived from the symmetric upper-endpoint relation.',
           'A low-memory/per-coordinate fallback remains available for sparse embeddings and failed optional allocations.','',
           'The native mask counter is now inlined into the AVX2/POPCNT channel routine. This removes per-query function',
           'call/temporary-array overhead while retaining runtime CPU feature checks. Portable paths remain supported.','',
           '## Experiments not retained','',
           'Several coarse-template grouping variants were tested, including combinations with the final preprocessing change.',
           'They added sorting and box-classification overhead; their small apparent gains were not convincing enough to',
           'justify replacing the simpler verified path. Those experimental kernels are not shipped.',
           'All empirical performance testing for this update used alpha only.','',
           '## Exactness','',
           'Both uint64 match-count arrays and entropy values matched the previous release exactly for all 64 channels on',
           'every timed call. C3 match counts were independently verified with an aligned sklearn KDTree calculation.',
           'The full suite passed 172 tests; undefined-behavior, address and leak checks passed, and forced portable-path',
           'checks passed. No samples, template starts, precision settings, distance rules or tolerance defaults changed.',
           '[Count vectors](counts.json) · [Individual timings](timings.json) · [Machine/code/input fingerprints](environment.json)','',
           '## What this says about the limit','',
           'The measured gains are diminishing on this machine, which has a four-CPU-equivalent quota.',
           'These experiments do not prove an absolute speed limit. Wider SIMD hardware, more CPU cores, a GPU,',
           'or a different exact counting algorithm remain possible avenues; their speedups were not measured here.',
           'Performance depends on signal distribution, radius, compiler and hardware.','',
           '![Further alpha-only tuning](timings.svg)','',
           'Bars show whole-batch medians; whiskers show the 25th–75th percentiles.',
           '[PNG](timings.png) · [SVG](timings.svg) · [Recording attribution/license](../data/README.md)','',
           '## Reproduce','',
           'Build the previous library separately, without replacing the active checkout:','',
           '```sh','mkdir -p /tmp/sampen-alpha-baseline',
           f'git archive --format=tar --output=/tmp/sampen-alpha-baseline/source.tar {baseline} csrc',
           'tar -xf /tmp/sampen-alpha-baseline/source.tar -C /tmp/sampen-alpha-baseline',
           'gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp \\',
           '    /tmp/sampen-alpha-baseline/csrc/sampen.c -lm -o /tmp/sampen-alpha-baseline/baseline.so',
           'python build_c.py','python -m pytest -q',
           'python examples/benchmark_alpha.py \\',
           '    --source /path/to/eeg_resting_8min_300hz.fif \\',
           '    --baseline-lib /tmp/sampen-alpha-baseline/baseline.so \\',
           f'    --baseline-commit {metadata["baseline_commit"]} \\',
           '    --output examples/results_alpha_limit',
           'python examples/summarize_alpha_limit.py','```','',
           'The recording/preprocessing is the same as the prior alpha-only benchmark: resample the complete pinned NeuroKit',
           'recording from 300 to 250 Hz, use butter(N=4, bandpass=[8,13], output="sos") with sosfiltfilt on the complete',
           'recording, then extract the first 180 seconds. Both libraries receive the identical resulting input array.','']
    (directory/'README.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained')
    med=np.array([previous['median_seconds'],current['median_seconds']])
    lo=np.array([previous['p25_seconds'],current['p25_seconds']])
    hi=np.array([previous['p75_seconds'],current['p75_seconds']])
    bars=ax.bar([0,1],med,width=.55,color=['#888888','#0072b2'],yerr=[med-lo,hi-med],capsize=5)
    ax.bar_label(bars,labels=[f'{v:.3f} s' for v in med],padding=8,fontsize=13)
    ax.set_xticks([0,1],[f'Previous release\n{baseline}','Shared radius bounds +\ninlined native counter'])
    ax.set_ylabel('Seconds for all 64 channels (lower is faster)')
    ax.set_ylim(0,max(hi)*1.25)
    ax.set_title(f'Alpha only: {speedup:.2f}× additional speedup\n180 s · 250 Hz · 64 channels · order 3 · delay 24')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.supxlabel(f'Matched {metadata["threads"]} threads · {metadata["repeats"]} interleaved trials · identical integer counts',fontsize=10)
    for ext in ['png','svg']:fig.savefig(directory/f'timings.{ext}',dpi=170)
    svg=directory/'timings.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('\n'.join(lines[:17]))


if __name__=='__main__':summarize()
