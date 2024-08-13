"""Implementation of the new v2 interface for COWs classes."""

from scipy.linalg import solve
from scipy.stats import uniform
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Tuple, Optional, Sequence, List, Dict, Callable, Any
from .typing import Density, FloatArray, Range
from .util import (
    pdf_from_histogram,
    fit_mixture,
    _quad_workaround,
    _get_pdf_parameters,
    _guess_starting_value,
    import_optional_module,
    make_weighted_negative_log_likelihood,
    FitError,
)
from .covariance import covariance_weighted_ml_fit
import re
import copy
import warnings
from functools import partial
from iminuit import Minuit
from iminuit.util import _normalize_limit


class CowsWarning(UserWarning):
    """Warning emitted by the Cows algorithm."""


class Cows:
    """
    Compute sWeights using COWs (new experimental API).

    This class replaces the old :class:`SWeight` and :class:`Cow` classes.
    It can compute classic sWeights and the generalized COWs. It automatically
    follows best practice based on the input that you give. When you use it
    wrong, it will complain.

    After initialization, the class instance is the weight function for the signal. To
    compute sWeights, you need to pass the sample in the discriminant variable. This
    sample must not be cut in any way, and the sWeights cannot be multiplied with other
    weights, unless these weights are independent of both the discriminant and the
    control variables.

    You can also get weight functions for each component, with the ``__getitem__``
    operator. You can iterate over the class instance to iterate over all weight
    functions. To get the number of weight functions, use ``len()`` on the instance.

    Furthermore, the class has some useful attributes, ``pdfs``, ``norm``, and
    ``yields``.
    """

    __slots__ = (
        "signal_pdfs",
        "background_pdfs",
        "norm",
        "yields",
        "minuit_discriminatory",
        "minuit_control",
        "_wm",
        "_am",
        "_init_kwargs",
    )

    signal_pdfs: List[Density]
    background_pdfs: List[Density]
    norm: Density
    yields: Optional[Sequence[float]]
    minuit_discriminatory: Optional[Minuit]
    minuit_control: Optional[Minuit]
    _wm: FloatArray
    _am: FloatArray
    _init_kwargs: Dict[str, Any]

    def __init__(
        self,
        x: Optional[ArrayLike],
        spdf: Union[Density, Sequence[Density]],
        bpdf: Union[Density, Sequence[Density]],
        norm: Optional[Union[Density, Tuple[FloatArray, FloatArray]]] = None,
        *,
        range: Optional[Range] = None,
        summation: Optional[bool] = None,
        yields: Optional[Sequence[float]] = None,
        bounds: Dict[Density, Dict[str, Range]] = {},
        starts: Dict[Density, Dict[str, float]] = {},
        validate: bool = True,
    ):
        """
        Initialize.

        Parameters
        ----------
        x: array-like or None
            Sample over the discriminant variable. You should pass this to compute COWs
            via summation. If ``sample`` is None, COWs are computed with via
            integration. COWs may be computed via integration also if we cannot
            guarantee that summation works correctly, as summation requires that
            ``norm`` is an unbiased estimate of the the total density. You can overwrite
            the automated choice via ``summation``.
        spdf: callable or sequence of callable
            Signal PDF in the discriminant variable. Must accept an array-like argument
            and return an array. The PDF must be normalized over the sample range. This
            can also be a sequence of PDFs that comprise the signal.
        bpdf : callable or sequence of callables
            Background PDF in the discriminant variable. Each must accept an array-like
            argument and return an array. The PDF must be normalized over the sample
            range. This can also be a sequence of PDFs that comprise the background.
        norm: callable or tuple(array, array) or None, optional (default is None)
            The normalization function in the COWs formula. If a callable is provided,
            it must accept an array-like argument and return an array. Passing a
            callable deactivates the summation method even if ``sample`` is provided.
            You can override this with ``summation=True``. If a tuple is provided, it
            must consist of two array, entries and bin edges of a histogram over the
            distriminant variable, in other words, the output of numpy.histogram. If
            this argument is None, the function is computed from the provided component
            PDFs and ``sample`` or ``yields``.
        range: tuple(float, float) or None, optional (default is None)
            Integration range to use if COWs are computed via integration. If range is
            None, and ``sample`` is not None, the range is computed from the sample.
        summation: bool or None, optional (default is None)
            If this is None, use summation (only possible if ``sample`` is set) unless
            the user provides an external normalization function with ``norm``, then we
            cannot guarantee that summation is possible. Setting this to True enforces
            summation and setting it to False enforces integration.
        yields: sequence of float or None, optional (default is None)
            If this is not None and ``norm`` is None, compute the normalization function
            from the component PDFs and these yields. This can be used to override the
            otherwise internally computed yields. If the PDFs are parametric, this
            argument is instead used as starting values for the fit.
        bounds: Dict[Density, Dict[str, Range]], optional (default is {})
            Allows to pass parameter bounds to the fitter for each density.
        starts: Dict[Density, Dict[str, float]], optional (default is {})
            Allows to pass parameter starting values to the fitter for each density.
        validate: bool, optional (default: True)
            Whether to validate internal fits with goodness-of-fit test. For
            getting optimal and unbiased weights, a linear combination of the component
            PDFs need to match the observed distribution. Using the summation method
            further requires that the ``norm`` function is an unbiased estimate of the
            observed distribution. If set to True, a goodness-of-fit
            test is performed to detect bad fits and a warning is emitted if the test
            fails. Otherwise, the test is skipped, which may slightly speed up the
            computation.

        Examples
        --------
        See :ref:`tutorials`.

        """
        spdf = list(spdf) if isinstance(spdf, Sequence) else [spdf]
        bpdf = list(bpdf) if isinstance(bpdf, Sequence) else [bpdf]

        self._init_kwargs = {
            "spdf": spdf,
            "bpdf": bpdf,
            "norm": norm,
            "range": range,
            "summation": summation,
            "yields": yields,
            "bounds": bounds,
            "starts": starts,
        }

        if x is not None:
            x = np.atleast_1d(x).astype(float)

        self.signal_pdfs = spdf
        self.background_pdfs = bpdf
        pdfs = self.signal_pdfs + self.background_pdfs

        if isinstance(norm, Sequence):
            xedges, self.norm = _process_histogram_argument(norm, range)
            range = xedges[0], xedges[-1]
        elif range is None:
            if x is None:
                raise ValueError(
                    "range must be set if x is None and norm is not a histogram"
                )
            range = (np.min(x), np.max(x))
            xedges = np.array(range)
        else:
            xedges = np.array(range)
        assert range is not None

        if isinstance(norm, Sequence):
            # already handled above
            assert self.norm is not None
        elif isinstance(norm, Density):
            if x is not None and summation is None:
                x = None
                warnings.warn(
                    "providing a x and an external function with norm "
                    "disables summation, override this with summation=True",
                    CowsWarning,
                    stacklevel=2,
                )
            self.norm = norm
        elif norm is None:
            has_parameters = any(_get_pdf_parameters(pdf) for pdf in pdfs)
            if x is None and yields is None:
                if has_parameters:
                    raise ValueError(
                        "x cannot be None if norm is None and "
                        "yields is None if pdfs have parameters"
                    )
                # no information about x or yields or norm and all pdfs fixed,
                # fall back to uniform norm, the only remaining possibility
                self.norm = uniform(range[0], range[1] - range[0]).pdf
            else:
                if x is not None and (yields is None or has_parameters):
                    # this overrides yields if they are set
                    nsig = len(self.signal_pdfs)
                    yields, pdfs, self.minuit_discriminatory = _fit_mixture(
                        x, pdfs, yields, bounds, starts, validate
                    )
                    self.signal_pdfs = pdfs[:nsig]
                    self.background_pdfs = pdfs[nsig:]
                assert yields is not None
                yields_sum = sum(yields, 0.0)

                def fn(x: FloatArray) -> FloatArray:
                    r = np.zeros_like(x)
                    for a, pdf in zip(yields, pdfs):
                        r += (a / yields_sum) * pdf(x)
                    return r

                self.norm = fn
        else:
            msg = (
                f"argument for norm ({type(norm)}) not recognized, "
                "see docs for valid types"
            )
            raise ValueError(msg)

        assert self.norm is not None

        if summation is False:
            x = None
        self._wm = _compute_lower_w_matrix(pdfs, self.norm, xedges, x)

        # invert W matrix to get A matrix using an algorithm
        # optimized for positive definite matrices
        self._am = solve(
            self._wm,
            np.identity(len(self)),
            lower=True,
            overwrite_a=True,
            overwrite_b=True,
            assume_a="pos",
        )
        self.yields = yields

    def __call__(self, x: FloatArray) -> FloatArray:
        """
        Compute weights for the signal component.

        Parameters
        ----------
        x: array of float
            Where in the domain of the discriminant variable to compute the weights.

        """
        return self._component("s")(x)

    def __getitem__(self, idx: Union[int, str]) -> Callable[[FloatArray], FloatArray]:
        """
        Return the weight function for the i-th component.

        You can also get the weight function for the signal and background by passing
        the strings 's' and 'b', respectively.
        """
        if isinstance(idx, str):
            if idx in "sb":
                return self._component(idx)
            else:
                msg = f"idx={idx!r} is not valid, use 's' or 'b'"
                raise ValueError(msg)
        if idx < 0:
            idx += len(self)
        if idx >= len(self):
            raise IndexError
        return self._component(idx)

    def __len__(self) -> int:
        """Return number of components."""
        return len(self.signal_pdfs) + len(self.background_pdfs)

    def _component(self, idx: Union[int, str]) -> Callable[[FloatArray], FloatArray]:
        # we compute only one pdf[k](x) array at a time, because x may be large
        norm = self.norm
        pdfs = self.signal_pdfs + self.background_pdfs
        if isinstance(idx, int):
            assert 0 <= idx < len(self)
            am = self._am[idx]
        elif idx == "s":
            am = np.sum(self._am[: len(self.signal_pdfs)], axis=0)
        else:
            am = np.sum(self._am[len(self.signal_pdfs) :], axis=0)

        def fn(x: FloatArray) -> FloatArray:
            nx = norm(x)
            nx[nx == 0] = np.nan
            w = np.zeros_like(x)
            for amk, pdf in zip(am, pdfs):
                w += amk * pdf(x)
            w /= nx
            return w

        return fn

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        model: Callable[..., FloatArray],
        *,
        bounds: Dict[str, Range] = {},
        starts: Dict[str, float] = {},
        covariance_estimation: str = "bootstrap",
        replicas: int = 100,
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Estimate model parameters in the  control sample.

        Parameters
        ----------
        x : array-like
            Sample over the discriminant variable. This must be the same that you used
            to compute COWs, if the COWs were estimated from data. The ``x`` and ``y``
            samples must form pairs.
        y : array-like
            Sample over the control variable, which correspond to the ``x`` sample. The
            order of x and y values must be matched so that they form pairs.
        model : callable
            Model that is fitted to the weighted sample. It is a callable that accepts
            the discriminatory variable as first argument, followed by the parameters
            that should be estimated as individual floats. Depending on the kind of fit
            (see ``method``), the model function should return the pdf, the cdf, or an
            integral, density pair.
        bounds: Dict[str, Range], optional (default is {})
            Allows to pass parameter bounds to the fitter for each density.
        starts: Dict[str, float], optional (default is {})
            Allows to pass parameter starting values to the fitter for each density.
        covariance_estimate: str, optional; default is "bootstrap"
            How to estimate the covariance matrix of the fitted parameters. If set to
            "bootstrap" (the default), full but slow error propagation is performed with
            the bootstrap method. A second option is "fast", which computes an
            approximation that neglects uncertainties in the COWs. This is often equal
            to full error propagation and must faster to compute. For the fastest
            results, use this together with a binned fit. We do not offer full
            analytical error propagation, because it is very complex to implement and
            also slow to compute. Use the bootstrap instead.
        replicas : int, optional; default is 100
            Number of bootstrap replicas that are computed. Only used if
            ``covariance_estimate`` is "bootstrap".

        Returns
        -------
        (values, covariance)
            A tuple consisting of an array with the best fit values, and another array
            that represents the covariance matrix.

        Notes
        -----
        We do not support extended maximum-likelihood fits for fitting the weighted
        sample in the control variable, because it makes no sense to fit a density. The
        signal yield should be computed by summing the weights (optionally also in bins
        of the control variable). An approximate uncertainty estimate per bin that
        neglects uncertainties in the COWs is sqrt(sum(w**2)). Full error propagation
        including bin-to-bin correlations can be achieved with the bootstrap.

        The internal bootstrap is using a pseudo-random number generator, which is
        seeded with the length of the input arrays, to make results from repeated calls
        deterministic.
        """
        x, y = np.atleast_1d(x, y)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        bootstrap = covariance_estimation == "bootstrap"
        w = self(x)
        self.minuit_control = _fit_weighted(y, w, model, bounds, starts, bootstrap)
        val = np.array(self.minuit_control.values[:])

        if bootstrap:
            covariance = import_optional_module("resample.bootstrap").covariance

            kwargs = copy.deepcopy(self._init_kwargs)
            dmin = self.minuit_discriminatory
            if dmin is not None:
                kwargs["yields"] = dmin.values[: len(self)]
                s: Dict[Any, Dict[str, float]] = {}
                pdfs = kwargs["spdf"] + kwargs["bpdf"]
                for name in dmin.parameters:
                    re_match = re.match(r"^pdf[(\d+)]:(.+)", name)
                    if re_match:
                        pdf = pdfs[int(re_match.group(1))]
                        pname = re_match.group(2)
                        pval = dmin.values[name]
                        if pdf in s:
                            s[pdf][pname] = pval
                        else:
                            s[pdf] = {pname: pval}
                kwargs["starts"] = s

            bounds = {}
            starts = {}
            min = self.minuit_control
            for par in min.parameters:
                bounds[par] = min.limits[par]
                starts[par] = min.values[par]

            def est(x: FloatArray, y: FloatArray) -> FloatArray:
                # replica fits tend to have slightly bad gof
                w = Cows(x, **kwargs, validate=False)(x)
                m = _fit_weighted(y, w, model, bounds, starts, bootstrap)
                return np.array(m.values[:])

            cov = covariance(
                est,
                x,
                y,
                method="extended",
                size=replicas,
                random_state=len(x),
            )
            cov = np.atleast_2d(cov)
        elif covariance_estimation == "fast":
            m = self.minuit_control
            cov = covariance_weighted_ml_fit(model, y, w, m.values[:], m.covariance)
        else:
            msg = f"unknown value covariance_estimation={covariance_estimation!r}"
            raise ValueError(msg)

        return val, cov


def _process_histogram_argument(
    arg: Tuple[FloatArray, FloatArray], range: Optional[Tuple[float, float]]
) -> Tuple[FloatArray, Density]:
    try:
        w, xe = arg
    except ValueError as e:
        e.args = (
            f"{e.args[0]} (histogram must be tuple of weights and bin edges, (w, xe))",
        )
        raise
    if len(w) != len(xe) - 1:
        msg = "counts and bin edges do not have the right lengths"
        raise ValueError(msg)
    if range is not None and (range[0] < xe[0] or xe[-1] < range[1]):
        msg = "sample range outside of histogram range"
        raise ValueError(msg)
    return xe, pdf_from_histogram(w, xe)


def _compute_lower_w_matrix(
    g: Sequence[Density],
    var: Density,
    xedges: FloatArray,
    sample: Optional[FloatArray],
) -> FloatArray:
    n = len(g)
    w = np.zeros((n, n))
    # only fill lower triangle
    for i in range(n):
        for j in range(i + 1):
            w[i, j] = _compute_w_element(g[i], g[j], var, xedges, sample)
    return w


def _compute_w_element(
    g1: Density,
    g2: Density,
    var: Density,
    xedges: FloatArray,
    sample: Optional[FloatArray],
) -> np.float64:
    if sample is None:

        def fn(m: FloatArray) -> FloatArray:
            return g1(m) * g2(m) / var(m)

        result = np.float64(0)
        for x0, x1 in zip(xedges[:-1], xedges[1:]):
            result += _quad_workaround(fn, x0, x1)

    else:
        g1x = g1(sample)
        g2x = g2(sample)
        varx = var(sample)
        result = np.mean(g1x * g2x * varx**-2)

    return result


def _fit_mixture(
    sample: FloatArray,
    pdfs: Sequence[Density],
    yields: Optional[Sequence[float]],
    bounds: Dict[Density, Dict[str, Range]],
    starts: Dict[Density, Dict[str, float]],
    validate: bool,
) -> Tuple[List[float], List[Density], Minuit]:
    fitted_pdfs: List[Density] = []
    yields, list_of_kwargs, minuit = fit_mixture(
        sample, pdfs, yields, bounds, starts, validate
    )
    for pdf, kwargs in zip(pdfs, list_of_kwargs):
        if kwargs:
            fitted_pdfs.append(partial(pdf, **kwargs))  # type:ignore
        else:
            fitted_pdfs.append(pdf)
    return yields, fitted_pdfs, minuit


def _fit_weighted(
    y: FloatArray,
    w: FloatArray,
    model: Callable[..., FloatArray],
    bounds: Dict[str, Range],
    starts: Dict[str, float],
    bootstrap: bool,
) -> Minuit:
    ## binned fit has problems, because sum of weights can be negative
    # val = np.histogram(y, bins=yedges, weights=w)[0]
    # var = np.histogram(y, bins=yedges, weights=w**2)[0]
    # wnll = BinnedNLL(np.transpose((val, var)), yedges, model)

    wnll = make_weighted_negative_log_likelihood(y, w, model)

    for k in wnll._parameters:  # type:ignore
        if k in bounds:
            wnll._parameters[k] = _normalize_limit(bounds[k])  # type:ignore

    for k, (a, b) in wnll._parameters.items():  # type:ignore
        if k not in starts:
            starts[k] = _guess_starting_value(a, b)

    min = Minuit(wnll, **starts)
    min.strategy = 0 if bootstrap else 1
    min.migrad()
    if not min.valid:
        msgs = ["fit failed", f"{min.fmin}", f"{min.params}"]
        raise FitError("\n".join(msgs))
    return min
