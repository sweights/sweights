import numpy as np
from sweights.independence import plot_indep_scatter, kendall_tau
from scipy.stats import chi2
from numpy.testing import assert_allclose


def test_kendall_tau():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=1000)
    y = rng.uniform(size=1000)
    val, err, pvalue = kendall_tau(x, y)
    assert pvalue > 0.01
    z = val / err
    pvalue2 = chi2(1).sf(z**2)
    assert_allclose(pvalue, pvalue2, atol=1e-3)


def test_plot_indep_scatter():
    x = np.random.uniform(size=1000)
    y = np.random.uniform(size=1000)
    plot_indep_scatter(x, y)
