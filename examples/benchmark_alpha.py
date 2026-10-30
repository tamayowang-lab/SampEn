"""Alpha-only interleaved comparison with the previous released C library.

Compile csrc from the selected previous commit separately, then pass
--baseline-lib and --baseline-commit (default baseline: 48b4202).
Both libraries are invoked through the same Python wrapper and thread count.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sampen_c
from prepare_resting_eeg import SHA256
from benchmark_180s import kdtree_counts


def previous_wrapper(path):
    lib=ctypes.CDLL(str(path.resolve()))
    for name in ['sampen_channels','sampen_channels_counts','sampen_has_openmp']:
        original=getattr(sampen_c._library(),name)
        getattr(lib,name).argtypes=original.argtypes
        getattr(lib,name).restype=original.restype
    spec=importlib.util.spec_from_file_location('previous_alpha_sampen',ROOT/'sampen_c.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._library=lambda:lib
    return module.sample_entropy_c


def benchmark(source,baseline,output,repeats=11,threads=4,baseline_commit='48b42029bc1d384e03cfae4d6d501f1325c8dc0e'):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Source recording SHA-256 mismatch')
    raw=mne.io.read_raw_fif(source,preload=False,verbose='ERROR')
    if raw.info['sfreq']!=300:
        raise ValueError('Unexpected source sample rate')
    names=['C3','Cz','O1']+[c for c in raw.copy().pick('eeg').ch_names if c not in ['C3','Cz','O1']]
    names=names[:64]
    full=resample_poly(raw.get_data(picks=names),5,6,axis=1)
    sos=butter(4,[8,13],btype='bandpass',fs=250,output='sos')
    x=np.ascontiguousarray(sosfiltfilt(sos,full,axis=1)[:,:45000])
    assert x.shape==(64,45000)
    previous=previous_wrapper(baseline)
    baseline_key=f'previous_release_{baseline_commit[:7]}'
    methods={
        baseline_key:lambda:previous(x,3,24,n_jobs=threads,return_counts=True),
        'optimized':lambda:sampen_c.sample_entropy_c(x,3,24,n_jobs=threads,return_counts=True),
    }
    expected=methods[baseline_key]()
    warmed=methods['optimized']()
    for a,b in zip(warmed,expected):np.testing.assert_array_equal(a,b)
    radius=.2*np.std(x[0])
    independent=kdtree_counts(x[0],24,radius)
    assert independent==[int(expected[1][0]),int(expected[2][0])]
    timings={method:[] for method in methods}
    rng=random.Random(813)
    for repeat in range(repeats):
        order=list(methods)
        rng.shuffle(order)
        for method in order:
            start=perf_counter()
            observed=methods[method]()
            timings[method].append(perf_counter()-start)
            for a,b in zip(observed,expected):np.testing.assert_array_equal(a,b)
        print(f'alpha repeat {repeat+1}/{repeats}',flush=True)
    summary={method:dict(median_seconds=float(np.median(t)),p25_seconds=float(np.percentile(t,25)),
                         p75_seconds=float(np.percentile(t,75)),min_seconds=min(t),max_seconds=max(t))
             for method,t in timings.items()}
    output.mkdir(parents=True,exist_ok=True)
    (output/'timings.json').write_text(json.dumps(dict(samples_seconds=timings,summary=summary,
        speedup=summary[baseline_key]['median_seconds']/summary['optimized']['median_seconds']),indent=2)+'\n')
    (output/'counts.json').write_text(json.dumps(dict(entropy=expected[0].tolist(),counts_m=expected[1].tolist(),
        counts_m_plus1=expected[2].tolist(),independent_C3_KDTree_counts=independent),indent=2)+'\n')
    cpuinfo=Path('/proc/cpuinfo')
    lines=cpuinfo.read_text().splitlines() if cpuinfo.exists() else []
    cpu=next((s.split(':',1)[1].strip() for s in lines if s.startswith('model name')),platform.processor())
    flags=next((s.split(':',1)[1].split() for s in lines if s.startswith('flags')),[])
    quota=Path('/sys/fs/cgroup/cpu.max')
    metadata=dict(source='NeuroKit resting EEG',source_sha256=SHA256,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  channels=names,sfreq=250,duration_seconds=180,samples=45000,order=3,delay_samples=24,
                  delay_seconds=.096,max_delay_seconds=.125,band_hz=[8,13],threads=threads,repeats=repeats,
                  baseline_commit=baseline_commit,
                  baseline_library_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256((ROOT/'_sampen_c.so').read_bytes()).hexdigest(),
                  code_sha256={name:hashlib.sha256((ROOT/'csrc'/name).read_bytes()).hexdigest()
                               for name in ['sampen.c','sampen.h','bitset_count.h','rank_count.h']},
                  cpu=cpu,avx2_available='avx2' in flags,popcnt_available='popcnt' in flags,
                  cpu_quota=quota.read_text().strip() if quota.exists() else 'unknown',python=sys.version,
                  platform=platform.platform(),versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','mne']},
                  preprocessing='Whole recording resample_poly(up=5,down=6); alpha 8–13 Hz butter(N=4,output=sos), '
                                'sosfiltfilt on complete recording; first 180 seconds afterward.',
                  tolerance='.2 * each filtered channel population SD',
                  timing='Whole batch through identical public Python wrapper, including validation, radius computation, '
                         'allocation, exact entropy and count output. Full-batch warm-up for both libraries. '
                         'Method order randomized each repeat, seed=813. Reading, preprocessing and warm-up excluded.',
                  correctness='Both uint64 count arrays and entropy checked with exact array equality on every call; '
                              'C3 counts independently checked with aligned KDTree.',
                  compiler='gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp; '
                           'AVX2/POPCNT only in runtime-dispatched functions, no fast-math or march=native',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--baseline-lib',type=Path,required=True)
    parser.add_argument('--baseline-commit',default='48b42029bc1d384e03cfae4d6d501f1325c8dc0e')
    parser.add_argument('--output',type=Path,default=ROOT/'examples/results_alpha')
    parser.add_argument('--repeats',type=int,default=11)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if min(args.repeats,args.threads)<1:parser.error('repeats and threads must be positive')
    benchmark(args.source,args.baseline_lib,args.output,args.repeats,args.threads,args.baseline_commit)
