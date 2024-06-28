"""Implementation of the COW class."""

from scipy.integrate import quad
from scipy import linalg
import numpy as np
from typing import Union, Tuple, Optional, Sequence, List
from .typing import Density, FloatArray
from .util import normalized, pdf_from_histogram
import warnings

__all__ = ["Cow"]


class Cow:
    """Produce weights using COWs."""

    mrange: Tuple[float, float]
    Im: Density
    gk: List[Density]
    Akl: FloatArray
    obs: Optional[Tuple[FloatArray, FloatArray]]
    renorm: bool

    def __init__(
        self,
        mrange: Tuple[float, float],
        gs: Density,
        gb: Union[Density, Sequence[Density]],
        Im: Union[int, Density] = 1,
        obs: Optional[Tuple[FloatArray, FloatArray]] = None,
        renorm: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize Cow object.

        This will compute the W and A (or alpha) matrices which are used to
        produce the weight functions. Evaluation of these functions on a
        dataset is done in a different function :func:`get_weight`.

        Parameters
        ----------
        mrange : sequence of two floats
            A sequence with two float elements which indicates the integration range in
            the discriminant variable.
        gs : callable
            The function for the signal pdf (numerator) must accept a single
            argument in this case the discriminant variable.
        gb : callable or sequence of callable
            A function or sequence of functions for the backgrond pdfs (numerator)
            which must each accept a single argument in this case the
            discriminant variable.
        Im : 1 or callable, optional
            The function for the "variance function" or I(m) (denominator)
            which must accept a single argument in this case the discriminant
            variable. Can also pass 1 for a uniform variance function (the
            default).
        obs : array-like, optional
            You can instead pass the observed distribution to evaluate Im
            instead. This expects the entries and bin edges in a two element
            tuple like the return value of np.histogram.
        renorm : bool, optional
            Renormalise passed functions to unity (you can override this if you
            already know it's true).
        verbose : bool, optional
            If True, produce extra output for debugging.

        Notes
        -----
        For more details see
        `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_

        See Also
        --------
        get_weight

        """
        self.renorm = renorm
        self.mrange = mrange

        if renorm:

            def normed(fn: Density) -> Density:
                return normalized(fn, mrange)

        else:

            def normed(fn: Density) -> Density:
                return fn

        self.gs = normed(gs)
        gbarg = [gb] if not isinstance(gb, Sequence) else gb
        self.gb = [normed(g) for g in gbarg]
        self.gk = [self.gs] + self.gb
        if isinstance(Im, int):
            self.Im: Density = lambda m: np.ones_like(m) / (mrange[1] - mrange[0])
        else:
            self.Im = normed(Im)
        self.obs = obs

        xe = np.array(mrange)
        if obs:
            try:
                w, xe = obs
            except IndexError:
                raise ValueError(
                    "The observation must be passed as length two object "
                    "containing weights and bin edges (w, xe) - ie. what is "
                    "returned by numpy.histogram()"
                )
            if len(w) != len(xe) - 1:
                msg = "The counts and bin edges do not have the right lengths."
                raise ValueError(msg)
            if xe[0] != mrange[0] or xe[-1] != mrange[1]:
                msg = "Histogram range does not match mrange"
                raise ValueError(mrange)
            self.Im = pdf_from_histogram(w, xe)

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

    def get_weight(self, k: int, m: FloatArray) -> FloatArray:
        """
        Return the weights.

        Parameters
        ----------
        k : int
            Index of the component
        m : ndarray
            Values of the discriminating variable to compute weights for

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
