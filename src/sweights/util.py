"""Various utilities used by the package or in the tutorials."""

from packaging.version import Version
from typing import Any
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.integrate import quad
import warnings

RooAbsPdf = Any
RooAbsReal = Any


def import_optional_module(name: str, *, min_version: str = ""):
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
    obs: RooAbsReal,
    *,
    npoints: int = 0,
    method: str = "makima",
    forcenorm: bool = True,
):
    """
    Convert a RooFit RooAbsPdf into a vectorized Python callable.

    This converts a RooFit::RooAbsPdf object into a Python callable that can be used by
    either the :class:`SWeight` or :class:`Cow` classes.

    Parameters
    ----------
    pdf : RooAbsPdf
        The pdf, must inherit from RooAbsPdf (e.g. RooGaussian, RooExponential,
        RooAddPdf etc.)
    obs : RooAbsReal
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
                # ignore accuracy warnings from quad
                warnings.simplefilter("ignore")
                norm = quad(fn, *range)[0]

            fn_orig = fn

            def fn(x):
                return fn_orig(x) / norm

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

        def fn(x):
            r = wrapper(x, pdf, obs)
            return np.array(r)

    return fn


def plot_binned(data, *, bins=None, range=None, weights=None, **kwargs):
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
    axes: Axes or None, optional
        Axes to plot on. If None, then use matplotlib.pyplot.gca().
    **kwargs:
        Further arguments are forwarded to matplotlib.pyplot.errorbar.
    """
    if weights is None:
        val, xe = np.histogram(data, bins=bins, range=range)
        err = val**0.5
    else:
        wsum, xe = np.histogram(data, bins=bins, range=range, weights=weights)
        w2sum = np.histogram(data, bins=xe, weights=weights**2)[0]
        val = wsum
        err = np.sqrt(w2sum)
    cx = 0.5 * (xe[1:] + xe[:-1])
    if "marker" in kwargs:
        kwargs["fmt"] = kwargs.pop("marker")
    elif "fmt" not in kwargs:
        kwargs["fmt"] = "o"
    plt = import_optional_module("matplotlib.pyplot")
    plt.errorbar(cx, val, err, **kwargs)
