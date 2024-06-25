import numpy as np
from sweights.independence import plot_indep_scatter


def test_plot_indep_scatter():
    x = np.random.uniform(size=1000)
    y = np.random.uniform(size=1000)
    plot_indep_scatter(x, y)
