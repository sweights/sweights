import numpy as np
from sweights.independence import plot_indep_scatter, kendall_tau


def test_kendall_tau():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=1000)
    y = rng.uniform(size=1000)
    val, err, pvalue = kendall_tau(x, y)
    print(val, err, pvalue)
    assert pvalue > 0.05


def test_plot_indep_scatter():
    x = np.random.uniform(size=1000)
    y = np.random.uniform(size=1000)
    plot_indep_scatter(x, y)
