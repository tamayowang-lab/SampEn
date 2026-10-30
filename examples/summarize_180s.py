"""Generate the report and exportable chart for the 180-second EEG benchmark."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def summarize(directory=ROOT/'results_180s'):
    rows=list(csv.DictReader((directory/'timings.csv').open()))
    metadata=json.loads((directory/'environment.json').read_text())
    groups={}
    for r in rows:groups.setdefault(r['band'],{})[r['method']]=r
    lines=['# Faster sample entropy for 180-second EEG segments','',
           'Input: **64 channels × 45,000 samples**, 250 Hz, order=3, tolerance=0.2 × each filtered channel SD.',
           f'Both methods use **{metadata["threads"]} threads** on the same cloud machine.',
           'Delays are round(250 / band midpoint), capped at floor(250 × 0.125) = **31 samples (124 ms)**.',
           'The generic entropy function accepts integer delays; this application-level cap is applied by the benchmark.',
           f'Timings are whole-batch medians of {metadata["repeats"]} calls; method order is randomized.',
           'File reading, resampling, band filtering and library/OpenMP warm-up are excluded.',
           'Input validation, radius calculation, allocations and complete entropy/count computation are included.','',
           '| Band | Hz | Delay (samples) | Delay (ms) | Previous C dual tree (s) | Optimized C auto (s) | Speedup |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for band,v in groups.items():
        old,new=v['previous_c_tree'],v['optimized_c_auto']
        a,b=float(old['median_seconds']),float(new['median_seconds'])
        lines.append(f'| {band} | {old["low_hz"]}–{old["high_hz"]} | {old["delay_samples"]} | '
                     f'{1000*float(old["delay_seconds"]):.0f} | {a:.3f} | {b:.3f} | **{a/b:.2f}×** |')
    lines+=['','## Exactness','',
            'The order-3 bitset method and the previous dual-tree method produced **identical uint64 pair counts**',
            'for both embedding dimensions on all 320 channel/band combinations. Entropy values also matched exactly',
            'in this run. Every timed call checked both count arrays and the entropy vector with exact array comparison.',
            'An independent aligned sklearn KDTree calculation additionally verified both C3 counts in every band.',
            'The raw [count vectors](counts.json) are retained. This is stronger than checking approximately equal logarithms.',
            'Cross-platform floating-point logarithms need not be bit-identical; no universal cross-platform claim is implied.','',
            '## Algorithm','',
            'Profiling the previous C code on the 180-second alpha C3 segment attributed about 96% of sampled runtime',
            'to recursive pair counting. The [profile](profile.txt) used GCC -O3 -pg -fno-inline; profiling timings are',
            'not the throughput timings reported above.','',
            'The new method reindexes templates by their first coordinate, so only a contiguous forward radius window',
            'must be checked. Prefix bitsets describe radius neighborhoods in the other coordinates. Partial blocks',
            'are corrected exactly with XOR; integer word intersections and population counts yield the same matches.',
            'POPCNT is selected only when the x86 CPU supports it; a tested portable path remains available.',
            'No approximate matching, template subsampling or fast-math is used.','',
            '`algorithm="auto"` selects bitsets for order=3 and 32,768–65,536 input samples, including this 45,000-sample workflow.',
            '`algorithm="bitset"` selects it explicitly; `algorithm="tree"` retains the previous method.',
            'Order=2 retains its range counter, and other automatic cases retain the dual tree.',
            'Prefix storage grows quadratically with a compact bit-packed constant; the automatic size limit bounds memory.',
            'At this length the bitset workspace is roughly 14 MiB per active channel, depending on delay.',
            'Worst-case runtime remains quadratic in signal length with word-level parallelism; these gains are measured, not an asymptotic guarantee.','',
            '## Recording and preprocessing','',
            'The pinned public NeuroKit recording was originally sampled at 300 Hz. It was resampled to 250 Hz with',
            '`scipy.signal.resample_poly(up=5, down=6)`, then filtered separately per band using',
            '`butter(N=4, btype="bandpass", output="sos")` and `sosfiltfilt`. Resampling and filtering used the complete',
            'eight-minute recording; the first 180 seconds were extracted afterward. No extra artifact rejection or',
            're-referencing was performed. Both methods receive exactly the same resulting arrays.',
            'See [source attribution and license](../data/README.md) and [full metadata](environment.json).',
            'This is one recording and one filter design, so timings may differ for your EEG and machine.','',
            '![180-second EEG timings](timings.svg)','',
            'Bars show whole-batch median runtimes; whiskers show the 25th–75th percentiles.',
            '[Raw timing table](timings.csv) · [PNG](timings.png) · [SVG](timings.svg)','',
            '## Reproduce','',
            'From the repository root, using the pinned source FIF linked on the data provenance page:','',
            '```sh','python build_c.py','python -m pytest -q',
            'python examples/benchmark_180s.py --source /path/to/eeg_resting_8min_300hz.fif',
            'python examples/summarize_180s.py','```','']
    (directory/'README.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(11.5,5),layout='constrained')
    xpos=np.arange(len(groups));width=.35
    old_values=[float(v['previous_c_tree']['median_seconds']) for v in groups.values()]
    new_values=[float(v['optimized_c_auto']['median_seconds']) for v in groups.values()]
    for offset,method,label,color in [(-width/2,'previous_c_tree','Previous C: dual tree','#888888'),
                                      (width/2,'optimized_c_auto','Optimized C: exact bitsets','#0072b2')]:
        r=[v[method] for v in groups.values()]
        med=np.array([float(a['median_seconds']) for a in r])
        lower=np.array([float(a['p25_seconds']) for a in r]);upper=np.array([float(a['p75_seconds']) for a in r])
        bars=ax.bar(xpos+offset,med,width,label=label,color=color,yerr=[med-lower,upper-med],capsize=3)
        ax.bar_label(bars,labels=[f'{t:.2f} s' for t in med],padding=5,fontsize=10)
    ax.set_xticks(xpos,[f'{band}\n{v["previous_c_tree"]["delay_samples"]} samples / '
                       f'{1000*float(v["previous_c_tree"]["delay_seconds"]):.0f} ms\n'
                       f'{a/b:.1f}× faster' for (band,v),a,b in zip(groups.items(),old_values,new_values)])
    ax.set_ylabel('Seconds for all 64 channels (lower is faster)')
    ax.set_title('180-second EEG · 250 Hz · 64 channels · order 3\nMatched 4-thread comparison; delay capped at 125 ms')
    ax.set_ylim(0,max(old_values)*1.25)
    ax.legend(loc='upper left',fontsize=10)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ext in ['png','svg']:fig.savefig(directory/f'timings.{ext}',dpi=170)
    svg=directory/'timings.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print('\n'.join(lines[:19]))


if __name__=='__main__':summarize()
