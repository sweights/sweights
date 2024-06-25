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
    calcualtion (the uncertainty calculation is trivial) which makes a
    few optimisations. See the scipy documentation for more information.
    """
    assert len(x) == len(y)
    err_approx = 1.0 / np.sqrt(len(x))
    return (kendalltau(x, y).correlation, err_approx, kendalltau(x, y).pvalue)


def plot_indep_scatter(x, y, reduction_factor=1, save=None, show=False):
    """
    Plot scatter of two variables.

    Plot a scatter graph of two variables and write the kendall tau
    coefficient.
    """
    plt = import_optional_module("matplotlib.pyplot")

    fig, ax = plt.subplots()
    ax.scatter(x[::reduction_factor], y[::reduction_factor], s=1)
    tau, err, pval = kendall_tau(x, y)
    ax.text(
        0.7,
        0.9,
        f"$\\tau = {tau} \\pm {err}$",
        transform=ax.transAxes,
        backgroundcolor="w",
    )
    ax.text(
        0.7,
        0.8,
        f"$p = {pval:.2f}$",
        transform=ax.transAxes,
        backgroundcolor="w",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show:
        plt.show()
    return fig, ax
