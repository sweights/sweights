"""Toy distributions to use in examples."""

import numpy as np
from typing import Tuple, Callable, Union, List
from numpy.typing import NDArray
from scipy.stats import norm, expon
from .typing import FloatArray


def make_classic_toy(
    seed: int,
    *,
    s: float = 5000,
    b: float = 5000,
    mrange: Tuple[float, float] = (0, 1),
    trange: Tuple[float, float] = (0, 1.5),
    ms_mu: float = 0.5,
    ms_sigma: float = 0.1,
    mb_mu: Union[float, Callable[[FloatArray], FloatArray]] = 0.5,
    ts_mu: float = 0.2,
    tb_mu: float = 0.1,
    tb_sigma: float = 0.1,
    random_sample_size: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a toy distribution to test the sweights and cows.

    This generates a classic toy in which the signal and background pdfs
    respectively factorize in the variables m and t.

    The signal pdf is a Gaussian in m and an exponential in t. The background pdf
    is an exponential in m and a Gaussian in t.

    Parameters
    ----------
    seed: int
        Seed for the random number generator.
    s: float, optional
        Expected yield of the signal. If ``random_sample_size`` is True, the actual
        number is randomly sampled around this value.
    b: float, optional
        Expected yield of the background. If argument ``random_sample_size`` is True,
        the actual number is randomly sampled around this value.
    mrange: (float, float), optional
        Range to which the m distribution is truncated.
    trange: (float, float), optional
        Range to which the t distribution is truncated.
    ms_mu: float, optional
        Expectation of the normal distribution in m. If the distribution is not
        truncated, this is the expectation of the untruncated distribution.
    ms_sigma: float, optional
        Standard deviation of the normal distribution in m. If the distribution is not
        truncated, this is the expectation of the untruncated distribution.
    mb_mu: float or callable, optional
        Expectation of the exponential distribution in m. If the distribution is not
        truncated, this is the expectation of the untruncated distribution. If it is a
        callable, it computes the value as a function of the t-variable. This can be
        used to test factorization violation.
    ts_mu: float, optional
        Expectation of the exponential distribution in m. If the distribution is not
        truncated, this is the expectation of the untruncated distribution.
    tb_mu: float, optional
        Expectation of the normal distribution in t. If the distribution is not
        truncated, this is the expectation of the untruncated distribution.
    tb_sigma: float, optional
        Standard deviation of the normal distribution in t. If the distribution is not
        truncated, this is the expectation of the untruncated distribution.
    random_sample_size: bool, optional
        If True, the yields are fluctuated around the given values according to the
        Poisson distribution.

    """
    rng = np.random.default_rng(seed)

    n_sig = rng.poisson(s) if random_sample_size else int(s)
    n_bkg = rng.poisson(b) if random_sample_size else int(b)

    dts = expon(0, ts_mu)
    dtb = norm(tb_mu, tb_sigma)

    t_s = dts.ppf(rng.uniform(*dts.cdf(trange), size=n_sig))  # type:ignore
    t_b = dtb.ppf(rng.uniform(*dtb.cdf(trange), size=n_bkg))  # type:ignore

    t = np.append(t_s, t_b)

    dms = norm(ms_mu, ms_sigma)
    m_s = dms.ppf(rng.uniform(*dms.cdf(mrange), size=n_sig))  # type:ignore

    m_b: Union[NDArray[np.float64], List[float]]
    if isinstance(mb_mu, (int, float)):
        dmb = expon(0, mb_mu)

        m_b = dmb.ppf(rng.uniform(*dmb.cdf(mrange), size=n_bkg))  # type:ignore
    else:
        m_b = []
        for mui in mb_mu(t_b):
            dmb = expon(0, mui)
            mi = dmb.ppf(rng.uniform(*dmb.cdf(mrange)))
            m_b.append(mi)

    m = np.append(m_s, m_b)

    sigmask = np.zeros(len(m), dtype=bool)
    sigmask[: len(m_s)] = 1

    perm = rng.permutation(np.arange(len(m)))
    m = m[perm]
    t = t[perm]
    sigmask = sigmask[perm]

    return m, t, sigmask
