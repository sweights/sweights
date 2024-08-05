"""Implementation of the new v2 interface for COWs classes."""

from scipy.linalg import solve
from scipy.stats import uniform
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Tuple, Optional, Sequence, List, Dict, Callable
from .typing import Density, FloatArray, Range
from .util import (
    pdf_from_histogram,
    fit_mixture,
    _quad_workaround,
    _get_pdf_parameters,
    FitValidation,
)
import warnings
from functools import partial


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

    __slots__ = ("pdfs", "norm", "yields", "_wm", "_am", "_sig")

    pdfs: List[Density]
    norm: Density
    yields: Optional[Sequence[float]]
    _wm: FloatArray
    _am: FloatArray
    _sig: int

    def __init__(
        self,
        sample: Optional[ArrayLike],
        spdf: Union[Density, Sequence[Density]],
        bpdf: Union[Density, Sequence[Density]],
        norm: Optional[Union[Density, Tuple[FloatArray, FloatArray]]] = None,
        *,
        range: Optional[Range] = None,
        summation: Optional[bool] = None,
        yields: Optional[Sequence[float]] = None,
        bounds: Dict[Density, Dict[str, Range]] = {},
        starts: Dict[Density, Dict[str, float]] = {},
        validation: FitValidation = FitValidation.GOF,
    ):
        """
        Initialize.

        Parameters
        ----------
        sample: array-like or None
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
        validation: FitValidation, optional (default: FitValidation.GOF)
            How to validate the internal fits, also see :class:`FitValidation`. For
            getting optimal and unbiased weights, a linear combination of the component
            PDFs need to match the observed distribution. Using the summation method
            further requires that the ``norm`` function is an unbiased estimate of the
            observed distribution. If set to DISPLAY, the full internal fit result is
            shown (displayed in a Jupyter notebook or printed on the terminal). If set
            to PLOT, only the fitted curve is plotted. If set to GOF, a goodness-of-fit
            test is performed to detect bad fits and a warning is emitted if the test
            fails. If you want nothing of that, you can slightly speed up the
            computation by skipping all of that with NONE.

        Examples
        --------
        See :ref:`tutorials`.

        """
        if sample is not None:
            sample = np.atleast_1d(sample).astype(float)

        spdfs = list(spdf) if isinstance(spdf, Sequence) else [spdf]
        bpdfs = list(bpdf) if isinstance(bpdf, Sequence) else [bpdf]
        self.pdfs = spdfs + bpdfs
        self._sig = len(spdfs)

        if isinstance(norm, Sequence):
            xedges, self.norm = _process_histogram_argument(norm, range)
            range = xedges[0], xedges[-1]
        elif range is None:
            if sample is None:
                raise ValueError(
                    "range must be set if sample is None and norm is not a histogram"
                )
            range = (np.min(sample), np.max(sample))
            xedges = np.array(range)
        else:
            xedges = np.array(range)
        assert range is not None

        if isinstance(norm, Sequence):
            # already handled above
            assert self.norm is not None
        elif isinstance(norm, Density):
            if sample is not None and summation is None:
                sample = None
                warnings.warn(
                    "providing a sample and an external function with norm "
                    "disables summation, override this with summation=True",
                    CowsWarning,
                    stacklevel=2,
                )
            self.norm = norm
        elif norm is None:
            has_parameters = any(_get_pdf_parameters(pdf) for pdf in self.pdfs)
            if sample is None and yields is None:
                if has_parameters:
                    raise ValueError(
                        "sample cannot be None if norm is None and "
                        "yields is None if pdfs have parameters"
                    )
                # no information about sample or yields or norm and all pdfs fixed,
                # fall back to uniform norm, the only remaining possibility
                self.norm = uniform(range[0], range[1] - range[0]).pdf
            else:
                if sample is not None and (yields is None or has_parameters):
                    # this overrides yields if they are set
                    yields, self.pdfs, nfit = _fit_mixture(
                        sample, self.pdfs, yields, bounds, starts, validation
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
            msg = (
                f"argument for norm ({type(norm)}) not recognized, "
                "see docs for valid types"
            )
            raise ValueError(msg)

        assert self.norm is not None

        if summation is False:
            sample = None
        self._wm = _compute_lower_w_matrix(self.pdfs, self.norm, xedges, sample)

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
        x: array of float or None, optional (default is None)
            Where in the domain of the discriminant variable to compute the weights. If
            the user already provided the sample of the discriminant variable during
            initialization, you can leave this to None and weights are computed for that
            sample.

        """
        return self._component(-1, x)

    def __getitem__(self, idx: Union[int, str]) -> Callable[[FloatArray], FloatArray]:
        """
        Return the weight function for the i-th component.

        You can also get the weight function for the signal and background by passing
        the strings 's' and 'b', respectively.
        """
        if isinstance(idx, str):
            if idx == "s":
                return lambda x: sum(  # type:ignore
                    self._component(i, x) for i in range(self._sig)
                )
            elif idx == "b":
                return lambda x: sum(  # type:ignore
                    self._component(i, x) for i in range(self._sig, len(self))
                )
            else:
                msg = f"idx={idx!r} is not valid, use 's' or 'b'"
                raise ValueError(msg)
        if idx < 0:
            idx += len(self)
        if idx >= len(self):
            raise IndexError
        return lambda x: self._component(idx, x)

    def __len__(self) -> int:
        """Return number of components."""
        return len(self.pdfs)

    def _component(self, idx: int, x: FloatArray) -> FloatArray:
        assert -1 <= idx < len(self)
        w = np.zeros_like(x)
        nx = self.norm(x)
        nx[nx == 0] = np.nan
        irange = range(self._sig) if idx < 0 else range(idx, idx + 1)
        for i in irange:
            # we compute only one pdf[k](x) array at a time, because x may be large
            for k, ak in enumerate(self._am[i]):
                w += ak * self.pdfs[k](x) / nx
        return w


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
    validate: FitValidation,
) -> Tuple[List[float], List[Density], int]:
    fitted_pdfs: List[Density] = []
    yields, list_of_kwargs = fit_mixture(sample, pdfs, yields, bounds, starts, validate)
    nfit = len(yields)
    for pdf, kwargs in zip(pdfs, list_of_kwargs):
        if kwargs:
            fitted_pdfs.append(partial(pdf, **kwargs))  # type:ignore
            nfit += len(kwargs)
        else:
            fitted_pdfs.append(pdf)
    return yields, fitted_pdfs, nfit
