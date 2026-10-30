import numpy as np
import pytest
from sklearn.neighbors import KDTree

from SampEn_utils import sample_entropy, sample_entropy_numba, _numba_sampen


def reference(x, order, delay, r):
    """Independent exhaustive pair count with equally extendable templates."""
    n = len(x) - order * delay
    b = a = 0
    for i in range(n):
        for j in range(i + 1, n):
            if all(abs(x[i+k*delay] - x[j+k*delay]) <= r for k in range(order)):
                b += 1
                a += abs(x[i+order*delay] - x[j+order*delay]) <= r
    return -np.log(a/b) if a and b else np.nan


@pytest.mark.parametrize('order', [2, 3, 5])
@pytest.mark.parametrize('delay', [1, 2, 3, 8])
@pytest.mark.parametrize('kind', ['normal', 'quantized', 'constant'])
def test_against_exhaustive(order, delay, kind):
    x = np.random.default_rng(72).normal(size=85)
    if kind == 'quantized':
        x = np.round(x)
    elif kind == 'constant':
        x[:] = 2
    for r in [0., 0.2, 1., 20.]:
        actual = sample_entropy_numba(x, order, delay, r)
        np.testing.assert_allclose(actual, reference(x, order, delay, r), atol=1e-14)


@pytest.mark.parametrize('delay', [1, 2, 5, 10])
def test_aligned_kdtree(delay):
    x = np.random.default_rng(12).normal(size=600)
    n = len(x) - 3*delay
    counts = []
    for m in [3, 4]:
        templates = x[np.arange(n)[:, None] + np.arange(m)*delay]
        counts.append(np.sum(KDTree(templates, metric="chebyshev").query_radius(templates, .7, count_only=True)-1))
    expected = -np.log(counts[1]/counts[0])
    np.testing.assert_allclose(sample_entropy_numba(x, 3, delay, .7), expected)


def test_original_functions_at_delay_one():
    x = np.random.default_rng(42).random(1000)
    r = .2*np.std(x)
    expected = sample_entropy(x)
    np.testing.assert_allclose(sample_entropy_numba(x), expected)
    np.testing.assert_allclose(_numba_sampen(x, 2, r), expected)


def test_layouts_and_inputs():
    x = np.random.default_rng(4).normal(size=400)[::2]
    x.flags.writeable = False
    expected = reference(x, 2, 3, .7)
    for values in [x, x.tolist()]:
        np.testing.assert_allclose(sample_entropy_numba(values, delay=3, tolerance=.7), expected)


def test_boundary_and_no_matches():
    # Quantized data deliberately place pairs exactly on the tolerance boundary.
    x = np.array([0., 1., 2., 0., 1., 2., 1., 0., 2.])
    np.testing.assert_allclose(sample_entropy_numba(x, tolerance=1), reference(x, 2, 1, 1))
    assert np.isnan(sample_entropy_numba(np.arange(10.), tolerance=0))
    assert sample_entropy_numba(np.ones(4), tolerance=0) == 0
    assert np.isnan(sample_entropy_numba([0, 0, 1, 1], tolerance=0))


@pytest.mark.parametrize('kwargs', [
    {'order': 1}, {'order': 2.5}, {'order': True}, {'delay': 0},
    {'delay': 1.5}, {'delay': False}, {'tolerance': -1},
    {'tolerance': np.inf}, {'tolerance': np.nan}, {'delay': 100},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        sample_entropy_numba(np.arange(20.), **kwargs)


@pytest.mark.parametrize('x', [[], [1, 2, 3], [[1, 2], [3, 4]], [0, 1, np.nan, 2], [0, 1, np.inf, 2], [0j]*5])
def test_invalid_signals(x):
    with pytest.raises(ValueError):
        sample_entropy_numba(x)
