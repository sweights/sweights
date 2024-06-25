"""Module to check and plot independence."""

import numpy as np
from scipy.stats import kendalltau
from .util import import_optional_module


def kendall_tau(x, y):
    """
    Return kendall tau correlation coefficient.

    Useful for ascertainting the extent to which two samples
    are independent. In particular for sweights and COWs one
    wants to know the extent to which the discriminant and
    control variables factorise.

    Parameters
    ----------
    x : ndarray
        Values in the first dimension - must have the same shape as `y`
    y : ndarray
        Values in the second dimension - must have the same shape as `x`

    Returns
    -------
    tuple :
        Two element tuple with the coefficient and
        its uncertainty

    Notes
    -----
    `x` and `y` must have the same dimension.
    This function now uses `scipy.stats.kendalltau` for the coefficent
    calculation (the uncertainty calculation is trivial) which makes a
    few optimisations. See the scipy documentation for more information.
    """
    assert len(x) == len(y)
    err_approx = 1.0 / np.sqrt(len(x))
    result = kendalltau(x, y)
    return (result.correlation, err_approx, result.pvalue)


def plot_indep_scatter(x, y, reduction_factor=None, save=None, show=False):
    """
    Plot scatter of two variables.

    Plot a scatter graph of two variables and write the kendall tau
    coefficient.
    """
    plt = import_optional_module("matplotlib.pyplot")

    fig, ax = plt.subplots()
    if reduction_factor is None:
        max_points = 5000
        if len(x) < max_points:
            reduction_factor = 1
        else:
            reduction_factor = len(x) // max_points
    ax.scatter(x[::reduction_factor], y[::reduction_factor], s=1)
    tau, err, pval = kendall_tau(x, y)
    ax.set_title(f"$\\tau = {tau:.3f} \\pm {err:.3f}$, p-value $= {pval:.2f}$")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show:
        plt.show()
    return fig, ax
