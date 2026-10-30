"""Exact 180-second, 250 Hz, 64-channel order-3 EEG benchmark.

Use the pinned NeuroKit FIF described in examples/data/README.md. Both C methods
receive identical band-filtered data. Every timed result checks exact integer
match counts, not just approximately equal entropy.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.neighbors import KDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sampen_c import sample_entropy_c
from prepare_resting_eeg import SHA256

BANDS=[('delta',.5,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]
SFREQ=250
DURATION=180
ORDER=3
MAX_DELAY_SECONDS=.125


def kdtree_counts(x,delay,radius):
    """Independent directed KDTree counts, converted to unordered pair counts."""
    n=len(x)-ORDER*delay
    result=[]
    for m in [ORDER,ORDER+1]:
        t=x[np.arange(n)[:,None]+np.arange(m)*delay]
        directed=int(KDTree(t,metric='chebyshev').query_radius(t,radius,count_only=True).sum())-n
        assert directed%2==0
        result.append(directed//2)
    return result


def benchmark(source,output,threads=4,repeats=3,channels=64):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Source recording SHA-256 mismatch')
    raw=mne.io.read_raw_fif(source,preload=False,verbose='ERROR')
    if raw.info['sfreq']!=300:
        raise ValueError('Unexpected source sampling rate')
    names=['C3','Cz','O1']+[c for c in raw.copy().pick('eeg').ch_names if c not in ['C3','Cz','O1']]
    names=names[:channels]
    if len(names)!=channels:
        raise ValueError('Not enough distinct EEG channels')
    # Resample the complete eight-minute recording before filtering and cropping.
    full=resample_poly(raw.get_data(picks=names),5,6,axis=1)
    output.mkdir(parents=True,exist_ok=True)
    rows,results=[],{}
    rng=random.Random(180)
    for band,low,high in BANDS:
        delay=min(round(SFREQ/((low+high)/2)),int(np.floor(SFREQ*MAX_DELAY_SECONDS)))
        sos=butter(4,[low,high],btype='bandpass',fs=SFREQ,output='sos')
        x=np.ascontiguousarray(sosfiltfilt(sos,full,axis=1)[:,:SFREQ*DURATION])
        assert x.shape==(channels,45000) and delay/SFREQ<=MAX_DELAY_SECONDS
        radius=.2*np.std(x,axis=1)
        # Warm library/OpenMP initialization and touch input outside timing.
        for algorithm in ['tree','bitset']:
            sample_entropy_c(x[:,:1024],ORDER,delay,n_jobs=threads,algorithm=algorithm)
        functions={
            'previous_c_tree':lambda:sample_entropy_c(x,ORDER,delay,n_jobs=threads,algorithm='tree',return_counts=True),
            'optimized_c_auto':lambda:sample_entropy_c(x,ORDER,delay,n_jobs=threads,return_counts=True),
        }
        print(f'{band}: {channels} x 45000, order=3, delay={delay} ({delay/SFREQ:.3f}s)',flush=True)
        expected=functions['previous_c_tree']()
        independent=kdtree_counts(x[0],delay,radius[0])
        assert independent==[int(expected[1][0]),int(expected[2][0])]
        timings={name:[] for name in functions}
        for repeat in range(repeats):
            methods=list(functions)
            rng.shuffle(methods)
            for method in methods:
                start=perf_counter()
                observed=functions[method]()
                timings[method].append(perf_counter()-start)
                for actual,reference in zip(observed,expected):
                    np.testing.assert_array_equal(actual,reference)
            print(f'  repeat {repeat+1}/{repeats}',flush=True)
        results[band]=dict(delay_samples=delay,delay_seconds=delay/SFREQ,radii=radius.tolist(),
                           entropy=expected[0].tolist(),counts_m=expected[1].tolist(),
                           counts_m_plus1=expected[2].tolist(),independent_C3_kdtree_counts=independent)
        for method,times in timings.items():
            row=dict(band=band,low_hz=low,high_hz=high,channels=channels,samples=45000,
                     order=ORDER,delay_samples=delay,delay_seconds=delay/SFREQ,threads=min(threads,channels),
                     method=method,median_seconds=float(np.median(times)),
                     p25_seconds=float(np.percentile(times,25)),p75_seconds=float(np.percentile(times,75)),
                     exact_counts_equal=True,entropy_equal=True)
            rows.append(row)
            print(f'  {method}: {row["median_seconds"]:.4f}s',flush=True)
        with (output/'timings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (output/'counts.json').write_text(json.dumps(results,indent=2)+'\n')
    cpuinfo=Path('/proc/cpuinfo')
    cpu=next((s.split(':',1)[1].strip() for s in cpuinfo.read_text().splitlines()
              if s.startswith('model name')),platform.processor()) if cpuinfo.exists() else platform.processor()
    quota=Path('/sys/fs/cgroup/cpu.max')
    metadata=dict(source='NeuroKit pinned 8-minute resting EEG',source_sha256=SHA256,
                  source_sfreq=300,sfreq=SFREQ,duration_seconds=DURATION,channels=names,
                  order=ORDER,max_delay_seconds=MAX_DELAY_SECONDS,threads=threads,repeats=repeats,
                  preprocessing='scipy.signal.resample_poly(up=5,down=6) on complete recording; '
                                '4th-order Butterworth bandpass, SOS zero-phase sosfiltfilt on complete recording; '
                                'first 180 seconds extracted afterward; no additional artifact rejection or re-referencing.',
                  tolerance='.2 * population SD of each filtered channel and segment',
                  timing='Entire channel batch, matched thread counts. Input validation, radius calculation, '
                         'allocations and all C computation included; reading, resampling, filtering and '
                         'library/OpenMP warm-up excluded. Randomized method order, seed=180.',
                  correctness='Every timed entropy and both uint64 match-count arrays equal previous C tree exactly. '
                              'C3 counts independently checked using aligned sklearn KDTree in every band.',
                  cpu=cpu,cpu_quota=quota.read_text().strip() if quota.exists() else 'unknown',
                  python=sys.version,platform=platform.platform(),
                  versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','mne']},
                  source_c_sha256=hashlib.sha256((ROOT/'csrc/sampen.c').read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256((ROOT/'_sampen_c.so').read_bytes()).hexdigest(),
                  header_sha256={name:hashlib.sha256((ROOT/'csrc'/name).read_bytes()).hexdigest()
                                 for name in ['sampen.h','bitset_count.h']},
                  compiler='gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp; no fast-math/march=native',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'examples/results_180s')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--channels',type=int,default=64)
    args=parser.parse_args()
    if min(args.threads,args.repeats,args.channels)<1:
        parser.error('threads, repeats and channels must be positive')
    benchmark(args.source,args.output,args.threads,args.repeats,args.channels)
