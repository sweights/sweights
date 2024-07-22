"""Module to check and plot independence."""

from scipy.stats import kendalltau
from .util import import_optional_module
from .typing import FloatArray
from typing import Tuple, Any, Optional

__all__ = ["kendall_tau", "plot_indep_scatter"]


def kendall_tau(x: FloatArray, y: FloatArray) -> Tuple[float, float, float]:
    """
    Return kendall tau correlation coefficient.

    Useful for ascertainting the extent to which two variables are independent and thus
    the PDF for both variables factorizes into two independent PDFs, one for each
    variable. This is a requirement to apply the classic sWeights method.

    WARNING: Using this function only makes sense if you have pure samples for all
    components considered in the sWeights method. You cannot apply this to a mixed
    sample. In general, you won't have these isolated samples for each component,
    because then you would not need sWeights. Yet, you can often get them from
    Monte-Carlo simulation of the experiment. If you trust the simulation, you can use
    this coefficient to test for factorization.

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
    result = kendalltau(x, y)
    n = len(x)
    # formula from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    approx_var = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    return (result.correlation, approx_var**0.5, result.pvalue)


def plot_indep_scatter(
    x: FloatArray,
    y: FloatArray,
    reduction_factor: int = 0,
    save: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """
    Plot scatter of two variables.

    Plot a scatter graph of two variables and write the kendall tau
    coefficient.
    """
    plt = import_optional_module("matplotlib.pyplot")

    fig, ax = plt.subplots()
    if reduction_factor == 0:
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
