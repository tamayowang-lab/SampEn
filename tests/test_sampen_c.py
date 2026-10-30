from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sampen_c import sample_entropy_c
from test_sample_entropy_numba import reference


@pytest.mark.parametrize('algorithm', ['tree', 'sweep'])
@pytest.mark.parametrize('order', [2, 3, 5])
@pytest.mark.parametrize('delay', [1, 2, 5, 8])
def test_c_against_exhaustive(algorithm, order, delay):
    rng = np.random.default_rng(14)
    x = np.vstack([rng.normal(size=85), rng.integers(-2, 3, size=85), np.ones(85), np.arange(85)])
    for r in [0., .5, 1., 50.]:
        expected = [reference(ch, order, delay, r) for ch in x]
        actual = sample_entropy_c(x, order, delay, r, algorithm=algorithm)
        np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize('algorithm', ['tree', 'sweep'])
def test_parallel_layout_and_channel_tolerances(algorithm):
    x = np.random.default_rng(42).normal(size=(8, 600))[:, ::2]
    x.flags.writeable = False
    radii = np.linspace(.3, 1., len(x))
    expected = [reference(ch, 2, 3, r) for ch, r in zip(x, radii)]
    for n_jobs in [1, 2, 4, -1]:
        np.testing.assert_allclose(sample_entropy_c(x, delay=3, tolerance=radii,
                                                  n_jobs=n_jobs, algorithm=algorithm), expected)
    np.testing.assert_allclose(sample_entropy_c(x, delay=3, algorithm=algorithm),
                               [reference(ch, 2, 3, .2*np.std(ch)) for ch in x], equal_nan=True)


@pytest.mark.parametrize('r', [np.nextafter(1., 0.), 1., np.nextafter(1., 2.)])
def test_radius_boundaries_and_ties(r):
    x = np.tile([0., 1., 2., 0., 1., 3., 2., 1., 0.], 8)[None, :]
    expected = [reference(x[0], 3, 2, r)]
    np.testing.assert_allclose(sample_entropy_c(x, 3, 2, r), expected, atol=1e-14)


def test_smallest_input_and_zero_matches():
    np.testing.assert_equal(sample_entropy_c([[1., 1., 1., 1.]], tolerance=0), [0.])
    assert np.isnan(sample_entropy_c([[0., 0., 1., 1.]], tolerance=0)[0])
    assert np.isnan(sample_entropy_c([np.arange(50.)], tolerance=0)[0])


def test_concurrent_independent_calls():
    x = np.random.default_rng(21).normal(size=(4, 500))
    expected = sample_entropy_c(x, delay=4, n_jobs=2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: sample_entropy_c(x, delay=4, n_jobs=2), range(6)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('kwargs', [
    {'order': 1}, {'order': 2.5}, {'order': True}, {'delay': 0}, {'delay': 1.5},
    {'delay': False}, {'delay': 1000}, {'tolerance': -1}, {'tolerance': np.inf},
    {'tolerance': np.nan}, {'tolerance': [1, 2, 3]}, {'tolerance': [[1]]},
    {'n_jobs': 0}, {'n_jobs': -2}, {'n_jobs': 1.5}, {'n_jobs': True},
    {'algorithm': 'approximate'},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        sample_entropy_c(np.ones((2, 20)), **kwargs)


@pytest.mark.parametrize('x', [[], [1, 2, 3, 4], np.empty((0, 20)), np.empty((2, 0)),
                              [[1, 2, 3]], [[0, np.nan, 1, 2]], [[0, np.inf, 1, 2]], [[0j]*5]])
def test_invalid_signals(x):
    with pytest.raises(ValueError):
        sample_entropy_c(x)


@pytest.mark.parametrize('delay', [1, 2, 5, 8])
@pytest.mark.parametrize('r', [0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 10.])
def test_range_counter(delay, r):
    rng = np.random.default_rng(135)
    x = np.vstack([rng.normal(size=180), rng.integers(-2, 3, size=180), np.ones(180)])
    expected = [reference(ch, 2, delay, r) for ch in x]
    for threads in [1, 3]:
        np.testing.assert_allclose(sample_entropy_c(x, delay=delay, tolerance=r,
                                                  algorithm='range', n_jobs=threads), expected, atol=1e-14)


def test_range_tree_agree_on_larger_input():
    x = np.random.default_rng(12).normal(size=(4, 4096))
    for delay in [1, 4, 15]:
        expected = sample_entropy_c(x, delay=delay, algorithm='tree')
        np.testing.assert_array_equal(sample_entropy_c(x, delay=delay, n_jobs=4), expected)
    with pytest.raises(ValueError):
        sample_entropy_c(x, order=3, algorithm='range')


@pytest.mark.parametrize('templates', [2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513])
def test_bitset_word_and_prefix_boundaries(templates):
    order,delay=3,7
    rng=np.random.default_rng(42)
    n=templates+order*delay
    x=np.vstack([rng.normal(size=n),rng.integers(-2,3,size=n),np.ones(n)])
    for radius in [0.,1.]:
        expected=[]
        for m in [order,order+1]:
            counts=[]
            for ch in x:
                t=ch[np.arange(templates)[:,None]+np.arange(m)*delay]
                distances=np.max(np.abs(t[:,None,:]-t[None,:,:]),axis=2)
                counts.append(np.count_nonzero(np.triu(distances<=radius,k=1)))
            expected.append(np.array(counts,dtype=np.uint64))
        for algorithm in ['tree','sweep','bitset']:
            result,b,a=sample_entropy_c(x,order,delay,radius,algorithm=algorithm,
                                       return_counts=True,n_jobs=3)
            np.testing.assert_array_equal(b,expected[0])
            np.testing.assert_array_equal(a,expected[1])


@pytest.mark.parametrize('order',[2,3,5])
@pytest.mark.parametrize('delay',[1,4,11])
def test_bitset_counts_match_other_algorithms(order,delay):
    x=np.random.default_rng(4).normal(size=(4,1000))[:,::2]
    x.flags.writeable=False
    radii=np.array([0.,.5,1.,100.])
    expected=sample_entropy_c(x,order,delay,radii,algorithm='tree',return_counts=True)
    for threads in [1,4]:
        for a,b in zip(sample_entropy_c(x,order,delay,radii,algorithm='bitset',
                                        n_jobs=threads,return_counts=True),expected):
            np.testing.assert_array_equal(a,b)
    if order==2:
        for a,b in zip(sample_entropy_c(x,order,delay,radii,algorithm='range',return_counts=True),expected):
            np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('r',[np.nextafter(1.,0.),1.,np.nextafter(1.,2.)])
def test_bitset_exact_radius(r):
    x=np.tile([0.,1.,2.,0.,1.,3.,2.,1.,0.],80)[None,:]
    expected=sample_entropy_c(x,3,2,r,algorithm='tree',return_counts=True)
    for a,b in zip(sample_entropy_c(x,3,2,r,algorithm='bitset',return_counts=True),expected):
        np.testing.assert_array_equal(a,b)


def test_return_counts_validation():
    with pytest.raises(ValueError):
        sample_entropy_c(np.ones((2,20)),return_counts='yes')


@pytest.mark.parametrize('templates',[32767,32768,32769,65535,65536,65537])
def test_rank_signed_boundary_and_simd_limit(templates):
    delay=7
    x=np.tile([0.,1.,2.,3.],(templates+3*delay+3)//4)[:templates+3*delay][None,:]
    expected=[]
    for m in [3,4]:
        t=x[0,np.arange(templates)[:,None]+np.arange(m)*delay]
        _,counts=np.unique(t,axis=0,return_counts=True)
        expected.append(sum(int(c)*(int(c)-1)//2 for c in counts))
    entropy,b,a=sample_entropy_c(x,3,delay,0,algorithm='bitset',return_counts=True)
    np.testing.assert_array_equal(b,[expected[0]])
    np.testing.assert_array_equal(a,[expected[1]])


def test_rank_continuous_large_counts():
    x=np.random.default_rng(51).normal(size=(1,65536+12))
    expected=sample_entropy_c(x,3,4,.5,algorithm='tree',return_counts=True)
    actual=sample_entropy_c(x,3,4,.5,algorithm='bitset',return_counts=True)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('radius',[0.,1.,np.finfo(float).max])
def test_radix_extremes_signed_zero_and_subnormals(radius):
    tiny=np.nextafter(0.,1.)
    values=[-np.finfo(float).max,-tiny,-0.,0.,tiny,np.finfo(float).max]
    x=np.tile(values,100)[None,:]
    expected=sample_entropy_c(x,3,7,radius,algorithm='tree',return_counts=True)
    observed=sample_entropy_c(x,3,7,radius,algorithm='bitset',return_counts=True)
    for a,b in zip(observed,expected):np.testing.assert_array_equal(a,b)
