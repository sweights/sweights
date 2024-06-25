from packaging.version import Version
from typing import Any, Optional
import warnings
import numpy as np

RooAbsPdf = Any
RooAbsReal = Any


def import_optional_module(name: str, min_version: str = ""):
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
            f" {name} is an optional dependency, "
            "please install it manually to use this function."
        )
        raise


def convert_rf_pdf(
    pdf: RooAbsPdf,
    obs: RooAbsReal,
    npoints: Optional[int] = None,
    forcenorm: bool = False,
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
    forcenorm : bool, optional
        Force the return function to be normalised by performing a numerical
        integration of it (the function should in most cases be normalised
        properly anyway so this shouldn't be needed)

    Returns
    -------
    callable :
        A callable function representing a normalised pdf which can then be
        passed to the :class:`SWeight` or :class:`Cow` classes

    """
    if npoints is not None:
        warnings.warn(
            "argument npoints is deprecated and does not do anything anymore",
            FutureWarning,
        )

    R = import_optional_module("ROOT", min_version="6")

    assert isinstance(obs, R.RooAbsReal)

    # range = (obs.getMin(), obs.getMax())

    # xvals = np.linspace(*range, npoints)
    # yvals = []

    # normset = R.RooArgSet(obs)
    # for x in xvals:
    #     obs.setVal(x)
    #     yvals.append(pdf.getVal(normset))

    # f = InterpolatedUnivariateSpline(xvals, yvals)

    # N = 1
    # if forcenorm:
    #     N = nquad(f, (range,))[0]

    # def retf(x):
    #     return f(x) / N

    # return retf

    pdf_norm = pdf.getNorm([obs]) if forcenorm else 1.0

    wrapper = getattr(R, "RooAbsPdfPythonWrapper", None)
    if wrapper is None:
        R.gInterpreter.Declare(
            f"""std::vector<double> RooAbsPdfPythonWrapper(
                const std::vector<double>& x, RooAbsPdf* pdf, RooRealVar* obs) {{
        std::vector<double> result;
        result.reserve(x.size());
        RooArgSet nset(*obs);
        for (const auto& xi : x) {{
            obs->setVal(xi);
            result.push_back(pdf->getVal(nset) / {pdf_norm});
        }}
        return result;
        }}"""
        )
        wrapper = getattr(R, "RooAbsPdfPythonWrapper")

    def fn(x):
        r = wrapper(x, pdf, obs)
        return np.array(r)

    return fn
