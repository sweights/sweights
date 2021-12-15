"""Implementation of the COW class."""

from scipy.stats import uniform
from scipy.integrate import quad
from scipy import linalg
import numpy as np


class Cow:
    """Produce weights using COWs."""

    def __init__(self, mrange, gs, gb, Im=1, obs=None, renorm=True, verbose=True):
        """
        Initialize Cow object.

        This will compute the W and A (or alpha) matrices which are used to
        produce the weight functions. Evaluation of these functions on a
        dataset is done in a different function :func:`get_weight`.

        Parameters
        ----------
        mrange : tuple
            A two element tuple for the integration range in the discriminant
            variable
        gs : callable
            The function for the signal pdf (numerator) must accept a single
            argument in this case the discriminant variable
        gb : callable or list of callable
            A function or list of functions for the backgrond pdfs (numerator)
            which must each accept a single argument in this case the
            discriminant variable
        Im : int or callable, optional
            The function for the "variance function" or I(m) (denominator)
            which must accept a single argument in this case the discriminant
            variable. Can also pass 1 for a uniform variance function (the
            default)
        obs : tuple of ndarray, optional
            You can instead pass the observed distribution to evaluate Im
            instead. This expects the entries and bin edges in a two element
            tuple like the return value of np.histogram
        renorm : bool, optional
            Renormalise passed functions to unity (you can override this if you
            already know it's true)

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
        self.gs = self._normalise(gs)
        self.gb = (
            [self._normalise(g) for g in gb]
            if hasattr(gb, "__iter__")
            else [self._normalise(gb)]
        )
        self.gk = [self.gs] + self.gb
        if Im == 1:
            un = uniform(*mrange)
            n = np.diff(un.cdf(mrange))
            self.Im = lambda m: un.pdf(m) / n
        else:
            self.Im = self._normalise(Im)

        self.obs = obs
        if obs:
            if len(obs) != 2:
                raise ValueError(
                    """The observation must be passed as length two object
                    containing weights and bin edges (w,xe) - ie. what is
                    returned by np.histogram()"""
                )
            w, xe = obs
            if len(w) != len(xe) - 1:
                raise ValueError(
                    """The bin edges and weights do not have the right
                    respective dimensions"""
                )
            # normalise
            w = w / np.sum(w)  # sum of wts now 1
            w /= (mrange[1] - mrange[0]) / len(
                w
            )  # divide by bin width to get a function which integrates to 1

            def f(m):
                return w[np.argmin(m >= xe) - 1]

            self.Im = np.vectorize(f)

        if verbose:
            print("Initialising COW:")

        # compute Wkl matrix
        self.Wkl = self._comp_Wkl()
        if verbose:
            print("    W-matrix:")
            print("\t" + str(self.Wkl).replace("\n", "\n\t "))

        # invert for Akl matrix
        self.Akl = linalg.solve(self.Wkl, np.identity(len(self.Wkl)), assume_a="pos")
        if verbose:
            print("    A-matrix:")
            print("\t" + str(self.Akl).replace("\n", "\n\t "))

    def _normalise(self, f):
        if self.renorm:
            N = quad(f, *self.mrange)[0]
            return lambda m: f(m) / N
        else:
            return f

    def _comp_Wkl_elem(self, k, j):

        # check it's available in m
        assert k < len(self.gk)
        assert j < len(self.gk)

        def integral(m):
            return self.gk[k](m) * self.gk[j](m) / self.Im(m)

        if self.obs is None:
            return quad(integral, self.mrange[0], self.mrange[1])[0]
        else:
            tint = 0
            xe = self.obs[1]
            for le, he in zip(xe[:-1], xe[1:]):
                tint += quad(integral, le, he)[0]
            return tint

    def _comp_Wkl(self):

        n = len(self.gk)

        ret = np.identity(n)

        for i in range(n):
            for j in range(n):
                if i > j:
                    ret[i, j] = ret[j, i]
                else:
                    ret[i, j] = self._comp_Wkl_elem(i, j)

        return ret

    def wk(self, k, m):
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
        n = len(self.gk)
        return np.sum(
            [self.Akl[k, j] * self.gk[j](m) / self.Im(m) for j in range(n)], axis=0
        )

    def get_weight(self, k, m):
        """
        Return the weights.

        Wrapper for `wk`

        See Also
        --------
        wk
        """
        return self.wk(k, m)
