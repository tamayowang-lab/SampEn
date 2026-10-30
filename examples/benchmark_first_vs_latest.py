"""Alpha-only comparison of the first corrected implementations and latest C.

All methods receive the same 64 x 45000 input and four threads. Archived source
files/libraries are built separately; the active checkout is never reset.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import mne
import numba
import numpy as np
from numba import njit, prange
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.neighbors import KDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sampen_c
from prepare_resting_eeg import SHA256

FIRST_NUMBA_COMMIT='76fa55f89b19b62820a6d80e3a3a23deec7efd12'
FIRST_C_COMMIT='4b559230e66a5cfe89a2969a8479f657daa96415'


def load_numba(path):
    global FIRST_KERNEL
    spec=importlib.util.spec_from_file_location('first_correct_numba',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    FIRST_KERNEL=module._numba_sampen_delay
    return module


@njit(parallel=True)
def _first_numba_channels(x,radii):
    out=np.empty(x.shape[0])
    for ch in prange(x.shape[0]):
        out[ch]=FIRST_KERNEL(x[ch],3,24,radii[ch])
    return out


def numba_channels(x):
    x=np.ascontiguousarray(x,dtype=np.float64)
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite input')
    return _first_numba_channels(x,.2*np.std(x,axis=1))


def first_c_wrapper(path):
    lib=ctypes.CDLL(str(path.resolve()))
    for name in ['sampen_channels','sampen_has_openmp']:
        current=getattr(sampen_c._library(),name)
        getattr(lib,name).argtypes=current.argtypes
        getattr(lib,name).restype=current.restype
    spec=importlib.util.spec_from_file_location('first_correct_c_wrapper',ROOT/'sampen_c.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module._library=lambda:lib
    return module.sample_entropy_c


def kd_row(x):
    n=len(x)-3*24
    radius=.2*np.std(x)
    counts=[]
    for order in [3,4]:
        t=x[np.arange(n)[:,None]+np.arange(order)*24]
        directed=int(KDTree(t,metric='chebyshev').query_radius(t,radius,count_only=True).sum())-n
        assert directed%2==0
        counts.append(directed//2)
    entropy=-np.log(counts[1]/counts[0]) if all(counts) else np.nan
    return entropy,counts[0],counts[1]


def run(source,first_c_lib,first_numba_file,output,repeats=5,threads=4):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Source recording SHA-256 mismatch')
    raw=mne.io.read_raw_fif(source,preload=False,verbose='ERROR')
    if raw.info['sfreq']!=300:raise ValueError('Unexpected sampling rate')
    names=['C3','Cz','O1']+[c for c in raw.copy().pick('eeg').ch_names if c not in ['C3','Cz','O1']]
    names=names[:64]
    full=resample_poly(raw.get_data(picks=names),5,6,axis=1)
    x=np.ascontiguousarray(sosfiltfilt(butter(4,[8,13],btype='bandpass',fs=250,output='sos'),full,axis=1)[:,:45000])
    assert x.shape==(64,45000)
    first_module=load_numba(first_numba_file)
    first_c=first_c_wrapper(first_c_lib)
    numba.set_num_threads(threads)
    # Import/JIT and thread-pool/library startup are excluded.
    warm=np.ascontiguousarray(x[:2,:512])
    numba_channels(warm)
    first_module.sample_entropy_numba(warm[0],order=3,delay=24)
    rng=random.Random(640180)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        reference=list(pool.map(kd_row,x))
        expected=np.array([r[0] for r in reference])
        expected_b=np.array([r[1] for r in reference],dtype=np.uint64)
        expected_a=np.array([r[2] for r in reference],dtype=np.uint64)
        latest,b,a=sampen_c.sample_entropy_c(x,3,24,n_jobs=threads,return_counts=True)
        np.testing.assert_array_equal(b,expected_b);np.testing.assert_array_equal(a,expected_a)
        np.testing.assert_allclose(latest,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
        methods={
            'aligned_kdtree':lambda:np.array([r[0] for r in pool.map(kd_row,x)]),
            'first_correct_numba_76fa55f':lambda:numba_channels(x),
            'first_correct_c_4b55923':lambda:first_c(x,3,24,n_jobs=threads,algorithm='tree'),
            'latest_c':lambda:sampen_c.sample_entropy_c(x,3,24,n_jobs=threads),
        }
        for name,fn in methods.items():
            print(f'Full-batch warm-up/validation: {name}',flush=True)
            np.testing.assert_allclose(fn(),expected,rtol=1e-12,atol=1e-12,equal_nan=True)
        timings={name:[] for name in methods}
        max_error={name:0. for name in methods}
        for repeat in range(repeats):
            order=list(methods);rng.shuffle(order)
            for name in order:
                start=perf_counter();observed=methods[name]();elapsed=perf_counter()-start
                np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
                finite=np.isfinite(expected)
                if finite.any():max_error[name]=max(max_error[name],float(np.max(np.abs(observed[finite]-expected[finite]))))
                timings[name].append(elapsed)
                print(f'Repeat {repeat+1}/{repeats}: {name} {elapsed:.4f}s',flush=True)
    summary={name:dict(median_seconds=float(np.median(t)),p25_seconds=float(np.percentile(t,25)),
                      p75_seconds=float(np.percentile(t,75)),min_seconds=min(t),max_seconds=max(t),
                      max_abs_entropy_error=max_error[name]) for name,t in timings.items()}
    output.mkdir(parents=True,exist_ok=True)
    (output/'timings.json').write_text(json.dumps(dict(summary=summary,samples_seconds=timings),indent=2)+'\n')
    (output/'counts.json').write_text(json.dumps(dict(entropy=expected.tolist(),counts_m=expected_b.tolist(),
        counts_m_plus1=expected_a.tolist(),latest_counts_exactly_equal_to_kdtree=True),indent=2)+'\n')
    cpuinfo=Path('/proc/cpuinfo');lines=cpuinfo.read_text().splitlines() if cpuinfo.exists() else []
    cpu=next((s.split(':',1)[1].strip() for s in lines if s.startswith('model name')),platform.processor())
    quota=Path('/sys/fs/cgroup/cpu.max')
    native_commit=subprocess.run(['git','log','-1','--format=%H','--','csrc'],cwd=ROOT,
                                 capture_output=True,text=True)
    metadata=dict(latest_native_commit=native_commit.stdout.strip() if native_commit.returncode==0 else None,
                  source_sha256=SHA256,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),channels=names,
                  sfreq=250,samples=45000,duration_seconds=180,band_hz=[8,13],order=3,delay_samples=24,
                  delay_seconds=.096,max_delay_seconds=.125,tolerance='.2 * each filtered channel population SD',
                  threads=threads,repeats=repeats,first_numba_commit=FIRST_NUMBA_COMMIT,first_c_commit=FIRST_C_COMMIT,
                  first_numba_source_sha256=hashlib.sha256(first_numba_file.read_bytes()).hexdigest(),
                  first_c_library_sha256=hashlib.sha256(first_c_lib.read_bytes()).hexdigest(),
                  latest_c_library_sha256=hashlib.sha256((ROOT/'_sampen_c.so').read_bytes()).hexdigest(),
                  latest_code_sha256={name:hashlib.sha256((ROOT/'csrc'/name).read_bytes()).hexdigest()
                                      for name in ['sampen.c','sampen.h','bitset_count.h','rank_count.h']},
                  first_numba_adapter='Unmodified archived _numba_sampen_delay kernel, called in a compiled prange channel loop; '
                                      'batch validation/radius calculation included. JIT warm-up excluded.',
                  first_c_adapter='Unmodified archived C binary, via current ctypes wrapper with algorithm=tree. '
                                  'Original C ABI has no integer-count output; entropy is validated.',
                  kdtree_adapter='Aligned first reference definition; two KDTree radius counts per channel; persistent thread pool.',
                  latest_c='Current compiled backend, algorithm=auto, entropy-only public call for timing.',
                  correctness='Every timed entropy vector allclose to aligned KDTree (rtol=atol=1e-12). '
                              'Latest uint64 counts equal independently computed KDTree counts for all 64 channels. '
                              'The historical Numba/C APIs return entropy only, so raw historical counts are not claimed.',
                  preprocessing='Complete pinned 300 Hz recording resampled with resample_poly(5,6), '
                                'butter(N=4,bandpass=[8,13],output=sos), sosfiltfilt on full recording, first 180 seconds afterward.',
                  timing='Whole channel batch; wrappers, validation, radius computation, allocations and computation included. '
                         'Reading, preprocessing, imports, JIT, and full-batch warm-up excluded. '
                         f'Randomized method order, seed=640180. All methods use the same {threads}-thread budget.',
                  cpu=cpu,cpu_quota=quota.read_text().strip() if quota.exists() else 'unknown',python=sys.version,
                  platform=platform.platform(),numba_threading_layer=numba.threading_layer(),
                  versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','numba','llvmlite','scikit-learn','mne']},
                  compiler='Both C libraries: gcc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror -fopenmp; no fast-math/march=native',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--first-c-lib',type=Path,required=True)
    parser.add_argument('--first-numba-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'examples/results_first_vs_latest')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if min(args.threads,args.repeats)<1:parser.error('threads and repeats must be positive')
    run(args.source,args.first_c_lib,args.first_numba_file,args.output,args.repeats,args.threads)
