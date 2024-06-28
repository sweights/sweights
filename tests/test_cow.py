from sweights.toy import make_classic_toy
from sweights.cow import Cow
from scipy.stats import norm, expon
from scipy.optimize import minimize
import numpy as np
from numpy.testing import assert_allclose


def test_cow():
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
        d = norm(ms_mu, ms_sigma)
        dnorm = np.diff(d.cdf(mrange))
        return d.pdf(m) / dnorm

    def gb(m):
        d = expon(0, mb_mu)
        dnorm = np.diff(d.cdf(mrange))
        return d.pdf(m) / dnorm

    cow = Cow(mrange, gs, gb, 1)

    w = cow.get_weight(0, m)

    def wnll(par):
        d = expon(0, *par)
        dnorm = np.diff(d.cdf(trange))
        return -2 * np.sum(w * (d.logpdf(t) - np.log(dnorm)))

    r = minimize(wnll, [1.0], bounds=[(1e-3, None)])
    assert r.success
    assert_allclose(r.x, ts_mu, atol=1e-4)
