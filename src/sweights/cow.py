"""Implementation of the COW class."""

from scipy.integrate import quad
from scipy import linalg
import numpy as np
from typing import Union, Tuple, Optional, Sequence, List, Any
from .typing import Density, FloatArray, Range
from .util import normalized, pdf_from_histogram
import warnings

__all__ = ["Cow"]


class Cow:
    """Produce weights using COWs."""

    mrange: Range
    Im: Density
    gk: List[Density]
    Akl: FloatArray
    renorm: bool
    ksig: int

    def __init__(
        self,
        mrange: Range,
        gs: Union[Density, Sequence[Density]],
        gb: Union[Density, Sequence[Density]],
        Im: Optional[Union[int, Density, Tuple[FloatArray, FloatArray]]] = None,
        renorm: bool = True,
        verbose: bool = False,
        **deprecated: Any,
    ):
        """
        Initialize Cow object.

        This will compute the W and A (or alpha) matrices which are used to
        produce the weight functions. Evaluation of these functions on a
        dataset is done in a different function :func:`get_weight`.

        Parameters
        ----------
        mrange : tuple(float, float)
            Integration range in the discriminant variable.
        gs : callable  or sequence of callables
            Signal PDF in the discriminant variable. Must accept an array-like argument
            and return an array.  This can also be a sequence of
            PDFs that comprise the signal.
        gb : callable or sequence of callables
            Background PDF in the discriminant variable. Each must accept an
            array-like argument and return an array. This can also be a sequence of
            PDFs that comprise the background.
        Im : callable or tuple(array-like, array-like) or None, optional
            The "variance function" in the COWs formula. An arbitrary density normalized
            over the integration range. If a callable is provided, it must accept an
            array-like argument and return an array. If a tuple is provide, it must
            consist of two array, entries and bin edges of a histogram over the
            distriminant variable, in other words, the output of numpy.histogram. If
            this argument is None, a constant density is used (the default).
        renorm : bool, optional
            Renormalise passed functions to unity (you can override this if you
            already know it's true).
        verbose : bool, optional
            If True, produce extra output for debugging.
        **deprecated :
            Deprecated arguments, which will raise a warning when used.

        Notes
        -----
        For more details see
        `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_

        See Also
        --------
        get_weight

        """
        DEPRECATED = {"obs"}
        unknown = set(deprecated) - DEPRECATED
        if unknown:
            msg = f"unknown arguments: {unknown}"
            raise KeyError(msg)

        self.renorm = renorm
        self.mrange = mrange

        if renorm:

            def normed(fn: Density) -> Density:
                return normalized(fn, mrange)

        else:

            def normed(fn: Density) -> Density:
                return fn

        gs1 = [gs] if not isinstance(gs, Sequence) else gs
        gs2 = [normed(g) for g in gs1]
        self.ksig = len(gs2)
        gb1 = [gb] if not isinstance(gb, Sequence) else gb
        gb2 = [normed(g) for g in gb1]
        self.gk = gs2 + gb2

        xe = np.array(mrange)
        if Im is None or isinstance(Im, int):
            if isinstance(Im, int):
                msg = "Passing Im=1 is deprecated, use Im=None instead"
                warnings.warn(msg, FutureWarning)
            self.Im: Density = lambda m: np.ones_like(m) / (mrange[1] - mrange[0])
        elif isinstance(Im, Sequence) or "obs" in deprecated:
            Im = deprecated.get("obs", Im)
            assert isinstance(Im, Sequence)
            xe = Im[1]
            self.Im = _process_histogram_argument(Im, mrange)
        else:
            self.Im = normed(Im)

        if verbose:
            print("Initialising COW:")

        self.Wkl = _compute_W(self.gk, self.Im, xe)
        if verbose:
            print("    W-matrix:")
            print("\t" + str(self.Wkl).replace("\n", "\n\t "))

        # invert for Akl matrix
        self.Akl = linalg.solve(self.Wkl, np.identity(len(self.Wkl)), assume_a="pos")
        if verbose:
            print("    A-matrix:")
            print("\t" + str(self.Akl).replace("\n", "\n\t "))

    def __call__(self, m: FloatArray) -> FloatArray:
        """
        Return signal weights.

        Parameters
        ----------
        m : ndarray
            Values of the discriminating variable to compute weights for.

        Returns
        -------
        ndarray :
            Values of the weights

        """
        return sum(self.get_weight(k, m) for k in range(self.ksig))  # type:ignore

    def get_weight(self, k: int, m: FloatArray) -> FloatArray:
        """
        Return weights for component k.

        Parameters
        ----------
        k : int
            Index of the component.
        m : ndarray
            Values of the discriminating variable to compute weights for.

        Returns
        -------
        ndarray :
            Values of the weights

        """
        im = self.Im(m)
        gm = [g(m) / im for g in self.gk]
        A = self.Akl[k]
        return A @ gm  # type:ignore

    # alias for get_weight
    wk = get_weight


def _process_histogram_argument(
    arg: Tuple[FloatArray, FloatArray], mrange: Range
) -> Density:
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
    if xe[0] != mrange[0] or xe[-1] != mrange[1]:
        msg = "histogram range does not match mrange"
        raise ValueError(mrange)
    return pdf_from_histogram(w, xe)


def _compute_W(
    gk: Sequence[Density],
    im: Density,
    me: FloatArray,
) -> FloatArray:
    n = len(gk)
    w = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                w[i, j] = w[j, i]
            else:
                w[i, j] = _compute_W_element(i, j, gk, im, me)
    return w


def _compute_W_element(
    k: int,
    j: int,
    gk: Sequence[Density],
    im: Density,
    me: Optional[FloatArray],
) -> float:
    def fn(m: FloatArray) -> FloatArray:
        return gk[k](m) * gk[j](m) / im(m)

    result = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for x0, x1 in zip(me[:-1], me[1:]):  # type:ignore
            result += quad(fn, x0, x1)[0]
    return result
