from sweights.testing import make_classic_toy
from sweights.experimental import Cows, CowsWarning
from scipy import stats
from scipy.optimize import minimize
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from typing import Callable


@pytest.mark.parametrize("norm_kind", ("fit", "yields", "g(m)", "hist"))
@pytest.mark.parametrize("sample_or_range_kind", ("sample", "range"))
def test_Cows(norm_kind, sample_or_range_kind):
    s = 10000
    b = 10000
    mrange = (0, 1)
    trange = (0, 1.5)
    ms_mu = 0.5
    ms_sigma = 0.1
    mb_mu = 0.5
    ts_mu = 0.2
    tb_mu = 0.1
    tb_sigma = 0.1

    m, t, sigmask = make_classic_toy(
        1,
        s=s,
        b=b,
        mrange=mrange,
        trange=trange,
        ms_mu=ms_mu,
        ms_sigma=ms_sigma,
        mb_mu=mb_mu,
        ts_mu=ts_mu,
        tb_mu=tb_mu,
        tb_sigma=tb_sigma,
    )

    def gs(m):
        d = stats.norm(ms_mu, ms_sigma)
        dnorm = np.diff(d.cdf(mrange))
        return d.pdf(m) / dnorm

    def gb(m):
        d = stats.expon(0, mb_mu)
        dnorm = np.diff(d.cdf(mrange))
        return d.pdf(m) / dnorm

    yields = None
    if norm_kind == "fit":
        norm = None
    elif norm_kind == "g(m)":

        def norm(m):
            return (gs(m) + gb(m)) / 2

    elif norm_kind == "yields":
        norm = None
        yields = (10.0, 10.0)
    else:
        norm = np.histogram(m, range=mrange)

    if sample_or_range_kind == "sample":
        sample = m
        range = None
    else:
        range = mrange
        sample = None

    if sample is None and norm is None:
        with pytest.raises(ValueError):
            Cows(sample, gs, gb, norm)
        return
    if sample is not None and isinstance(norm, Callable):
        with pytest.warns(CowsWarning):
            cows = Cows(sample, gs, gb, norm, yields=yields, range=range)
    else:
        cows = Cows(sample, gs, gb, norm, yields=yields, range=range)

    w = cows(m)
    assert_equal(w, cows(m, 0))

    def wnll(par):
        d = stats.expon(0, *par)
        dnorm = np.diff(d.cdf(trange))
        return -2 * np.sum(w * (d.logpdf(t) - np.log(dnorm)))

    r = minimize(wnll, [1.0], bounds=[(1e-3, None)])
    assert r.success
    assert_allclose(r.x, ts_mu, atol=1e-2)
