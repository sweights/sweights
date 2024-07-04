"""Various utilities used by the package or in the tutorials."""

from packaging.version import Version
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.integrate import quad
from scipy.special import comb
import warnings
from typing import Tuple, Optional, Union, Any, TYPE_CHECKING, List, Callable
from .typing import RooAbsPdf, RooRealVar, Density, FloatArray, Range
from numpy.typing import ArrayLike

__all__ = [
    "import_optional_module",
    "convert_rf_pdf",
    "plot_binned",
    "normalized",
    "pdf_from_histogram",
    "BernsteinBasisPdf",
    "make_bernstein_pdf",
    "make_weighted_negative_log_likelihood",
]


def import_optional_module(name: str, *, min_version: str = "") -> Any:
    """
    Import an optional dependency.

    In this package, we use optional dependencies in some places. We enhance the
    standard error message to make it more helpful and optionally check whether the
    version of the package matches the required minimum.
    """
    from importlib import import_module

    try:
        mod = import_module(name)

        if min_version:
            version = getattr(mod, "__version__", "0")

            if not Version(min_version) <= Version(version):
                msg = (
                    f"{name} found, but does not match the "
                    f"required minimum version{min_version}"
                )
                raise ImportError(msg)

        return mod

    except ModuleNotFoundError as e:
        e.msg += (
            " This is an optional dependency, "
            "please install it manually to use this function."
        )
        raise


def convert_rf_pdf(
    pdf: RooAbsPdf,
    obs: RooRealVar,
    *,
    npoints: int = 0,
    method: str = "makima",
    forcenorm: bool = True,
) -> Density:
    """
    Convert a RooFit RooAbsPdf into a vectorized Python callable.

    This converts a RooFit::RooAbsPdf object into a Python callable that can be used by
    either the :class:`SWeight` or :class:`Cow` classes.

    Parameters
    ----------
    pdf : RooAbsPdf
        The pdf, must inherit from RooAbsPdf (e.g. RooGaussian, RooExponential,
        RooAddPdf etc.)
    obs : RooRealVar
        The observable.
    npoints: int, optional (default is 0)
        If npoints is zero, a wrapper around the RooFit PDF is returned. This wrapper
        internally calls the RooFit PDF and therefore produces exact results. If npoints
        is larger than zero, a spline interpolator is constructed instead to approximate
        the RooFit PDF. It is constructed by evaluating the exact PDF at the given
        number of points, which are equally spaced over the range of ``obs``, which must
        be a bounded variable.  The spline is an approximation, but faster to compute.
    method: str, optional (default is "makima")
        Interpolation method to use. Accepted values are "makima" and "pchip".
    forcenorm : bool, optional (default is True)
        This only has an effect, if an interpolator is returned. Since the interpolator
        is not normalized in general, we need to normalize it. Setting this to False
        skips the computation of the integral, which is done numerically. Deactivate
        this only if the numerical interaction fails for some reason.

    Returns
    -------
    callable :
        A callable function representing a normalised pdf which can then be
        passed to the :class:`SWeight` or :class:`Cow` classes

    """
    R = import_optional_module("ROOT", min_version="6")

    assert isinstance(obs, R.RooAbsReal)

    range = (obs.getMin(), obs.getMax())

    if npoints > 0:
        x = np.linspace(*range, npoints)
        y = []

        for xi in x:
            obs.setVal(xi)
            y.append(pdf.getVal([obs]))

        # We only allow makima and pchip interpolators, because these do not overshoot.
        # This guarantees that the interpolators do not return negative values.
        fn: Density
        if method == "makima":
            fn = Akima1DInterpolator(x, y, method="makima")
        elif method == "pchip":
            fn = PchipInterpolator(x, y)
        else:
            msg = (
                f"method='{method}' is not recognized, "
                "allowed values are 'makima' and 'pchip'"
            )
            raise ValueError(msg)

        if forcenorm:
            with warnings.catch_warnings():
                # ignore accuracy warnings from integration
                warnings.simplefilter("ignore")
                return normalized(fn, range)

    else:
        wrapper = getattr(R, "RooAbsPdfPythonWrapper", None)
        if wrapper is None:
            R.gInterpreter.Declare(
                """std::vector<double> RooAbsPdfPythonWrapper(
                    const std::vector<double>& x, RooAbsPdf* pdf, RooRealVar* obs) {{
            std::vector<double> result;
            result.reserve(x.size());
            RooArgSet nset(*obs);
            for (const auto& xi : x) {{
                obs->setVal(xi);
                result.push_back(pdf->getVal(nset));
            }}
            return result;
}}"""
            )
            wrapper = getattr(R, "RooAbsPdfPythonWrapper")

        def fn(x: FloatArray) -> FloatArray:
            r = wrapper(x, pdf, obs)
            return np.array(r)

    return fn


def plot_binned(
    data: ArrayLike,
    *,
    bins: Optional[Union[int, FloatArray]] = 100,
    range: Optional[Range] = None,
    weights: Optional[ArrayLike] = None,
    density: bool = False,
    **kwargs: Any,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Plot histogram from data.

    Parameters
    ----------
    data: array-like
        Data to sort into a histogram.
    bins: int or array-like or None, optional
        Number of bins of the histogram.
    range: (float, float) or None, optional
        Range of the histogram.
    weights: array-like or None, optional
        Weights.
    density: bool, optional (default is False)
        If True, normalize the histogram.
    axes: Axes or None, optional
        Axes to plot on. If None, then use matplotlib.pyplot.gca().
    **kwargs:
        Further arguments are forwarded to matplotlib.pyplot.errorbar.

    """
    if TYPE_CHECKING:
        # workaround for buggy histogram annotations
        assert bins is not None
    if weights is None:
        val, xe = np.histogram(data, bins=bins, range=range)
        err = val**0.5
    else:
        wsum, xe = np.histogram(data, bins=bins, range=range, weights=weights)
        w2sum = np.histogram(data, bins=xe, weights=np.asarray(weights) ** 2)[0]
        val = wsum
        err = np.sqrt(w2sum)
    cx = 0.5 * (xe[1:] + xe[:-1])
    if "marker" in kwargs:
        kwargs["fmt"] = kwargs.pop("marker")
    elif "fmt" not in kwargs:
        kwargs["fmt"] = "o"
    if density:
        f = 1 / (np.sum(val) * np.diff(xe))
        val = val * f
        err = err * f
    plt = import_optional_module("matplotlib.pyplot")
    plt.errorbar(cx, val, err, **kwargs)
    return val, err, xe


def normalized(fn: Density, range: Range) -> Density:
    """Return a function normalized over the given range."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fnorm = quad(fn, *range)[0]
    return lambda x: fn(x) / fnorm


def pdf_from_histogram(w: FloatArray, xe: FloatArray) -> Density:
    """
    Return a pdf (piecewise constant) constructe from a histogram.

    Parameters
    ----------
    w: array of float or int
        Counts of the histogram.
    xe: array of float
        Edges of the histogram.

    Returns
    -------
    Callable
        A normalized density.

    """
    w = w / np.sum(w)
    w /= np.diff(xe)

    def fn(x: FloatArray) -> FloatArray:
        r = np.zeros_like(x)
        ind = np.searchsorted(xe, x, side="right") - 1
        mask = (0 <= ind) & (ind < len(xe) - 1)
        r[mask] = w[ind[mask]]
        return r

    return fn


class BernsteinBasisPdf:
    """Bernstein basis PDF."""

    def __init__(self, k: int, n: int, a: float, b: float):
        """
        Initialize Bernstein basis.

        Parameters
        ----------
        k: int
            Index of the basis.
        n: int
            Order of the polynom.
        a: float
            Starting value where the polynomial is defined and normalized.
        b: float
            Ending value where the polynomial is defined and normalized.

        """
        self._a = a
        self._iw = 1.0 / (b - a)
        self._fnorm: float = comb(n, k) * self._iw * (n + 1)
        self._k = k
        self._ak = n - k

    def __call__(self, x: FloatArray) -> FloatArray:
        """Compute probability density."""
        z: FloatArray = (x - self._a) * self._iw
        az: FloatArray = 1.0 - z
        return z**self._k * az**self._ak * self._fnorm


def make_bernstein_pdf(degree: int, a: float, b: float) -> List[Density]:
    """
    Construct a normalized Bernstein basis of given degree.

    The Bernstein basis polynomials returned by this function
    are normalized over the given range.

    Parameters
    ----------
    degree: int
        Order of the Bernstein basis. Must be at least 1.
    a: float
        Starting value where the polynomial is defined and normalized.
    b: float
        Ending value where the polynomial is defined and normalized.

    Returns
    -------
    Callable
        Normalized density.

    """
    if degree < 1:
        raise ValueError("minimum order is 1")

    return [BernsteinBasisPdf(i, degree, a, b) for i in range(degree + 1)]


def make_weighted_negative_log_likelihood(
    x: FloatArray,
    weights: FloatArray,
    model: Callable[..., FloatArray],
) -> Callable[..., float]:
    """Construct weighted log-likelihood function compatible with iminuit."""
    util = import_optional_module("iminuit.util")
    cost = import_optional_module("iminuit.cost")
    safe_log = cost._safe_log

    parameters = {}
    first = True
    for par, limits in util.describe(model, annotations=True).items():
        if first:
            first = False
            continue
        parameters[par] = limits

    def nll(*args: float) -> float:
        lp = safe_log(model(x, *args))
        return -2 * np.sum(weights * lp)  # type:ignore

    nll.errordef = 1.0  # type:ignore
    nll._parameters = parameters  # type:ignore

    return nll
