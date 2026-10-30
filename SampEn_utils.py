import numpy as np
from numba import jit, njit, types
from scipy.signal import periodogram, welch
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from math import factorial, log

def sample_entropy(x, order=2, delay=1, tolerance=None, metric="chebyshev"):
    """Sample Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    delay : int
        Spacing between template coordinates, in samples. Default is 1.
    tolerance : float
        Tolerance value for acceptance of the template vector. Default is 0.2
        times the standard deviation of x.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.

    Returns
    -------
    se : float
        Sample Entropy.

    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.

    The sample entropy of a signal :math:`x` is defined as:

    .. math:: H(x, m, r) = -\\log\\frac{C(m + 1, r)}{C(m, r)}

    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a
    `Chebyshev distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    inferior to :math:`r` and :math:`C(m, r)` is the number of embedded
    vectors of length :math:`m` having a Chebyshev distance inferior to
    :math:`r`.

    The KDTree code derives from mne-features by Jean-Baptiste Schiratti and
    Alexandre Gramfort. This implementation always uses sklearn's KDTree.
    For a Numba
    implementation supporting arbitrary delay and Chebyshev distance, use
    :func:`sample_entropy_numba`. The original denominator removes only one
    template regardless of delay; the Numba function instead aligns the two
    template sets. Consequently, results can differ when delay is greater than 1.

    References
    ----------
    Richman, J. S. et al. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    Examples
    --------
    Fractional Gaussian noise with H = 0.5

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x, order=2):.4f}")
    2.1819

    Same with order = 3 and using the Euclidean distance

    >>> print(f"{ant.sample_entropy(x, order=3, metric='euclidean'):.4f}")
    2.6806

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    1.9078

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    2.0555

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.sample_entropy(rng.random(1000)):.4f}")
    2.2017

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    0.1633

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    -0.0000
    """
    # define r

    if tolerance is None:
        r = 0.2 * np.std(x, ddof=0)
    else:
        assert isinstance(tolerance, (float, int))
        r = tolerance
        
    x = np.asarray(x, dtype=np.float64)
    # if metric == "chebyshev" and x.size < 5000:
    #     return _numba_sampen(x, order=order, r=r)
    # else: 
    phi = _app_samp_entropy(x, order=order, r=r, delay=delay, metric=metric, approximate=False)
    
    if phi[0][0] == 0 or phi[1][0] == 0:
        result = np.nan
    else:
        result = -np.log(np.divide(phi[1][0], phi[0][0]))
            
    return result
    
    
    
def _app_samp_entropy(x, order, r, delay=1, metric="chebyshev", approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`."""
    _all_metrics = KDTree.valid_metrics
    _all_metrics = _all_metrics() if callable(_all_metrics) else _all_metrics
    if metric not in _all_metrics:
        raise ValueError(
            "The given metric (%s) is not valid. The valid "
            "metric names are: %s" % (metric, _all_metrics)
        )
    phi = [[np.nan], [np.nan]]

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, delay)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
        
    count1 = (
        KDTree(emb_data1, metric=metric)
        .query_radius(emb_data1, r, count_only=True)
        .astype(np.float64)
    )
    # tree1 = cKDTree(emb_data1)
    # count1 = tree1.query_ball_point(emb_data1, r, p=np.inf, return_length=True).astype(np.float64)
    
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, delay)
    count2 = (
        KDTree(emb_data2, metric=metric)
        .query_radius(emb_data2, r, count_only=True)
        .astype(np.float64)
    )
    # tree2 = cKDTree(emb_data2)
    # count2 = tree2.query_ball_point(emb_data2, r, p=np.inf, return_length=True).astype(np.float64)
    
    if approximate:
        phi[0] = [np.mean(np.log(count1 / emb_data1.shape[0])), count1]
        phi[1] = [np.mean(np.log(count2 / emb_data2.shape[0])), count2]
    else:
        phi[0] = [np.mean((count1 - 1) / (emb_data1.shape[0] - 1)), count1]
        phi[1] = [np.mean((count2 - 1) / (emb_data2.shape[0] - 1)), count2]
    return phi


@jit(
    (types.Array(types.float64, 1, "C", readonly=True), types.int32, types.float64),
    nopython=True,
)
def _numba_sampen(sequence, order, r):
    """
    Fast evaluation of the sample entropy using Numba.
    """

    size = sequence.size
    # sequence = sequence.tolist()

    numerator = 0
    denominator = 0

    for offset in range(1, size - order):
        n_numerator = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= r
            n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            in_diff = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0:
        return 0  # use 0/0 == 0
    elif numerator == 0:
        return np.inf
    else:
        return -log(numerator / denominator)
    
def _embed(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : array_like
        1D-array of shape (n_times) or 2D-array of shape (signal_indice, n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.

    Returns
    -------
    embedded : array_like
        Embedded time series, of shape (..., n_times - (order - 1) * delay, order)
    """
    x = np.asarray(x)
    N = x.shape[-1]
    assert x.ndim in [1, 2], "Only 1D or 2D arrays are currently supported."
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")

    if x.ndim == 1:
        # 1D array (n_times)
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
        return Y.T
    else:
        # 2D array (signal_indice, n_times)
        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)
        embed_signal_length = N - (order - 1) * delay
        # define the new signal length
        indice = [[(i * delay), (i * delay + embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal
        for i in range(order):
            # loop with the order
            temp = x[:, indice[i][0] : indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)
            Y.append(temp)
            # append the sliced signal to list
        Y = np.concatenate(Y, axis=-1)
        return Y


def sample_entropy_numba(x, order=2, delay=1, tolerance=None):
    """Compute delay-aware sample entropy with Numba (Chebyshev distance).

    Both template sets use the same ``len(x) - order * delay`` start indices;
    an order-m template can therefore always be extended to order m+1. Every
    possible start is included, not just every ``delay``-th sample. Self matches
    are excluded and distances equal to the tolerance are included. As in
    :func:`sample_entropy`, return NaN if either match count is zero.

    Unlike the original KDTree function, which drops just one order-m template,
    this uses aligned template sets for delays greater than one. The functions
    agree at delay=1, except for floating point rounding.

    Sorting on the first template coordinate prunes pairs outside the radius.
    Remaining coordinates are checked with early rejection; no pairwise distance
    matrix is allocated. Worst-case runtime remains quadratic, with linear
    auxiliary memory. The first call includes JIT compilation; later calls reuse
    the compiled specialization (also cached on disk).

    Parameters
    ----------
    x : array_like
        Finite, real, one-dimensional signal.
    order : int
        Embedding dimension, at least 2.
    delay : int
        Positive spacing between coordinates, in samples.
    tolerance : float or None
        Nonnegative radius. None uses 0.2 times the population standard deviation.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("order must be an integer of at least 2")
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay must be a positive integer")
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError("x must be real")
    x = np.ascontiguousarray(raw, dtype=np.float64)
    if raw.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional signal")
    if int(order) * int(delay) > x.size - 2:
        raise ValueError("at least two extendable templates are required")
    r = 0.2 * np.std(x, ddof=0) if tolerance is None else float(tolerance)
    if not np.isfinite(r) or r < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if np.all(x == x[0]):
        return 0.0
    return _numba_sampen_delay(x, int(order), int(delay), r)


@njit(cache=True)
def _numba_sampen_delay(x, order, delay, r):
    """Count unordered, aligned template pairs with a sorted radius sweep."""
    n = x.size - order * delay
    indices = np.argsort(x[:n])
    denominator = 0
    numerator = 0
    for left in range(n - 1):
        i = indices[left]
        for right in range(left + 1, n):
            j = indices[right]
            if x[j] - x[i] > r:
                break
            matched = True
            for k in range(1, order):
                if abs(x[i + k * delay] - x[j + k * delay]) > r:
                    matched = False
                    break
            if matched:
                denominator += 1
                if abs(x[i + order * delay] - x[j + order * delay]) <= r:
                    numerator += 1
    if denominator == 0 or numerator == 0:
        return np.nan
    return -np.log(numerator / denominator)
