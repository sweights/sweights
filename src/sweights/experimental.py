"""Implementation of the new v2 interface for COWs and sWeights classes."""

from scipy.linalg import solve
import numpy as np
from typing import Union, Tuple, Optional, Sequence, List, Dict
from .typing import Density, FloatArray, Range
from .util import (
    pdf_from_histogram,
    fit_mixture,
    get_pdf_parameters,
    _quad_workaround,
    gof_pvalue,
    GofWarning,
)
import warnings
from functools import partial


class CowsWarning(UserWarning):
    """Warning emitted by the Cows algorithm."""


class Cows:
    """Compute weights using COWs."""

    pdfs: List[Density]
    var: Density
    yields: Optional[Sequence[float]]
    _sample: Optional[FloatArray]
    _am: FloatArray
    _sig: int

    def __init__(
        self,
        sample: Optional[FloatArray],
        spdf: Union[Density, Sequence[Density]],
        bpdf: Union[Density, Sequence[Density]],
        norm: Optional[Union[Density, Tuple[FloatArray, FloatArray]]] = None,
        *,
        range: Optional[Range] = None,
        summation: Optional[bool] = None,
        yields: Optional[Sequence[float]] = None,
        bounds: Dict[Density, Dict[str, Range]] = {},
        starts: Dict[Density, Dict[str, float]] = {},
        validate_input: bool = True,
    ):
        """
        Initialize.

        Parameters
        ----------
        sample: array or None
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
        validate_input: bool, optional (default is True)
            Whether to validate the input with a goodness-of-fit test. Applying COWs
            requires that the component PDFs indeed describe the observed distribution.
            Using the summation method further requires that the ``norm`` function is an
            unbiased estimate of the observed distribution. The goodness-of-fit test is
            able to detect violations of these requirements and emits a warning if the
            test fails. You can speed up the computation by setting this to False and
            skip the test.

        """
        self._sample = sample

        if sample is not None and range is None:
            range = (np.min(sample), np.max(sample))  # type:ignore

        spdfs = list(spdf) if isinstance(spdf, Sequence) else [spdf]
        self._sig = len(spdfs)
        bpdfs = list(bpdf) if isinstance(bpdf, Sequence) else [bpdf]
        self.pdfs = spdfs + bpdfs

        nfit = -1
        if isinstance(norm, Sequence):
            xedges, self.norm = _process_histogram_argument(norm, range)
        elif isinstance(norm, Density):
            if sample is not None and summation is None:
                sample = None
                warnings.warn(
                    "providing a sample and an external function with norm "
                    "disables summation, override this with summation=True",
                    CowsWarning,
                )
            xedges = np.array(range)
            self.norm = norm
            nfit = 0
        elif norm is None:
            if sample is None and yields is None:
                raise ValueError(
                    "norm cannot be None if sample is None and yields is None"
                )
            xedges = np.array(range)
            has_parameters = any(get_pdf_parameters(pdf) for pdf in self.pdfs)
            if yields is None or has_parameters:
                if sample is None:
                    msg = (
                        "sample cannot be None if norm is None and pdfs have parameters"
                    )
                    raise ValueError(msg)
                yields, self.pdfs, nfit = _fit_mixture(
                    sample, self.pdfs, yields, bounds, starts
                )
            assert yields is not None
            yields_sum = sum(yields, 0.0)

            def fn(x: FloatArray) -> FloatArray:
                r = np.zeros_like(x)
                for a, pdf in zip(yields, self.pdfs):
                    r += (a / yields_sum) * pdf(x)
                return r

            self.norm = fn
        else:
            msg = f"var type {type(norm)} not recognized, see docs for valid types"
            raise ValueError(msg)

        if xedges is None:
            raise ValueError(
                "if sample is None and norm is not a histogram, range must be set"
            )

        if summation is False:
            sample = None

        # If sample is not None here, summation technique is used to compute the W
        # matrix, which requires that norm is an estimate of the total pdf. We test this
        # with a GoF. No test is performed if norm derived from a histogram.
        if validate_input and sample is not None and nfit >= 0:
            assert len(xedges) == 2  # required by numerical integration in gof_pvalue
            pgof = gof_pvalue(sample, self.norm, nfit)
            if pgof < 0.01:
                msg = (
                    f"goodness-of-fit test produces small p-value ({pgof:.2g}), "
                    "check fit result"
                )
                warnings.warn(msg, GofWarning)

        # we must pass sample here, not self._sample, because sample may be set to None
        w = _compute_lower_w_matrix(self.pdfs, self.norm, xedges, sample)

        # invert W matrix to get A matrix using an algorithm
        # optimized for positive definite matrices
        self._am = solve(
            w,
            np.identity(len(w)),
            lower=True,
            overwrite_a=True,
            overwrite_b=True,
            assume_a="pos",
        )
        self.yields = yields

    def component(self, idx: int, x: Optional[FloatArray] = None) -> FloatArray:
        """
        Return weights for the indexed component.

        Parameters
        ----------
        idx: int
            Index of the component. If this is -1, compute the weights for the sum of
            signal components, if there are several.
        x: array of float or None, optional (default is None)
            Where in the domain of the discriminant variable to compute the weights. If
            the user already provided the sample of the discriminant variable during
            initialization, you can leave this to None and weights are computed for that
            sample.

        """
        if idx >= len(self.pdfs):
            raise IndexError("idx is out of bounds")

        if x is None:
            x = self._sample
        if x is None:
            raise ValueError("x cannot be None")

        w = np.zeros_like(x)
        nx = self.norm(x)
        nx[nx == 0] = np.nan
        irange = range(self._sig) if idx < 0 else range(idx, idx + 1)
        for i in irange:
            # we compute only one pdf[k](x) array at a time, because x may be large
            for k, ak in enumerate(self._am[i]):
                w += ak * self.pdfs[k](x) / nx
        return w

    def __call__(self, x: FloatArray, idx: int = -1) -> FloatArray:
        """Return weights for argument, see component(...) for details."""
        return self.component(idx, x)


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
) -> Tuple[List[float], List[Density], int]:
    fitted_pdfs: List[Density] = []
    yields, list_of_kwargs = fit_mixture(sample, pdfs, yields, bounds, starts)
    nfit = len(yields)
    for pdf, kwargs in zip(pdfs, list_of_kwargs):
        if kwargs:
            fitted_pdfs.append(partial(pdf, **kwargs))
            nfit += len(kwargs)
        else:
            fitted_pdfs.append(pdf)
    return yields, fitted_pdfs, nfit
