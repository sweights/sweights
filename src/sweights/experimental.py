"""Implementation of the new v2 interface for COWs and sWeights classes."""

from scipy.integrate import quad
from scipy.linalg import solve
import numpy as np
from typing import Union, Tuple, Optional, Sequence, List
from .typing import Density, FloatArray
from .util import pdf_from_histogram, fit_mixture, FitError
import warnings


class CowsWarning(UserWarning):
    """Warning emitted by the Cows algorithm."""


class Cows:
    """Compute weights using COWs."""

    pdfs: List[Density]
    var: Density
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
        range: Optional[Tuple[float, float]] = None,
        summation: Optional[bool] = None,
        yields: Optional[Sequence[float]] = None,
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
            ``norm`` is the total density. You can overwrite the automated choice via
            ``summation``.
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
            If this is None, use summation if possible unless the user provides an
            external normalization function with ``norm``, then we cannot guarantee that
            summation is possible. Setting this to True enforces summation and setting
            it to False enforces integration.
        yields: sequence of float or None, optional (default is None)
            If this is not None and ``norm`` is None, compute the normalization function
            from the component PDFs and these yields. This can be used to override the
            otherwise internally computed yields.

        """
        self._sample = sample

        if sample is not None and range is None:
            range = (np.min(sample), np.max(sample))  # type:ignore

        spdfs = list(spdf) if isinstance(spdf, Sequence) else [spdf]
        self._sig = len(spdfs)
        bpdfs = list(bpdf) if isinstance(bpdf, Sequence) else [bpdf]
        self.pdfs = spdfs + bpdfs

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
        elif norm is None:
            xedges = np.array(range)
            if yields is None:
                if sample is None:
                    raise ValueError(
                        "norm cannot be None if sample is None and yields is None"
                    )
                try:
                    yields = fit_mixture(sample, self.pdfs)  # type:ignore
                except FitError as e:
                    e.args = (e.args[0] + "; provide norm manually",)
                    raise

            assert yields is not None
            yields_sum: float = sum(yields)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for x0, x1 in zip(xedges[:-1], xedges[1:]):
                result += quad(fn, x0, x1)[0]

    else:
        g1x = g1(sample)
        g2x = g2(sample)
        varx = var(sample)
        result = np.mean(g1x * g2x * varx**-2)

    return result
