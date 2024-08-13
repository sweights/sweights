"""Implementation of covariance correction for weighted fits."""

import numpy as np
from .util import point_derivative
from typing import Callable, Any
from .typing import FloatArray

__all__ = ["covariance_weighted_ml_fit", "cov_correct"]


def covariance_weighted_ml_fit(
    pdf: Callable[..., FloatArray],
    x: FloatArray,
    w: FloatArray,
    val: FloatArray,
    cov: FloatArray,
    **deprecated: Any,
) -> FloatArray:
    """
    Compute the covariance matrix for an unbinned weighted maximum-likelihood fit.

    The computation is based on the sandwich estimator, which is asymptotically correct
    for an unbinned weighted likelihood fit of the form Q = -2 sum(w * log(pdf(x, p1,
    ..., pN))), where w are weights and x are observations, and p1 to pN are model
    parameters. Because of the weights, the standard computation via inversion of the
    Hesse matrix of Q is no longer correct. Instead one has to use the more general
    sandwich estimator, which is more expensive to calculate due to

    Parameters
    ----------
    pdf : callable
        The pdf model, which must take arguments (x, p1,...,pN), which has been fitted
        to the weighted sample. x is the observable and p1 to pN are shape parameters.
    x : array
        Sample of observations to which the pdf was fitted.
    w : array
        Weights used in the weighted maximum-likelihood fit.
    val : array-like
        The fitted parameter values.
    cov : array
        The false covariance matrix of the weighted likelihood fit as returned by
        Minuit, being equal to the inverse of the Hesse matrix of the weighted
        log-likelihood times -2.

    Returns
    -------
    array
        The corrected covariance matrix

    Notes
    -----
    This function corresponds to the weighted score function with independent
    weights (first term of Eq.51 in
    `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_).
    This is often a good approximation for sweights although will be an
    overestimate. For a better correction for sweights use `cov_correct` (which
    is only in general accurate when the shape parameters in the discriminating
    variable are known)

    See Also
    --------
    cov_correct

    """
    val = np.atleast_1d(val)
    cov = np.atleast_2d(cov)

    dim = len(val)
    if cov.shape != (dim, dim):
        msg = f"cov has wrong shape {cov.shape}, should be ({dim}, {dim})"
        raise ValueError(msg)

    deriv = point_derivative(pdf, x, val, 1e-3 * np.diag(cov) ** 0.5)
    v = np.empty(cov.shape)
    f = w**2 / pdf(x, *val) ** 2
    for i in range(len(val)):
        for k in range(i + 1):
            v[i, k] = np.sum(f * deriv[i] * deriv[k])
            if i != k:
                v[k, i] = v[i, k]
    return np.einsum("ij,jk,kl", cov, v, cov)


def cov_correct(
    hs, gxs, hdata, gdata, weights, Nxs, fvals, fcov, dw_dW_fs, Wvals, verbose=False
):
    """
    Perform a second order covariance correction for a fit to weighted data.

    Parameters
    ----------
    hs : callable
        The control variable pdf which must take arguments (x, p0,...,pn)
        which has been fitted to a weighted dataset, where x is the observable
        and p0 ... pn are the shape parameters
    gxs : list of callable
        A list of the disciminant variable pdfs which must take a single
        argument (x), where x is the observable (shape parameters of gxs must
        be known in this case)
    hdata : array
        The data values of the control variable observable at which the `hs`
        pdf is evaluated
    gdata : array
        The data values of the disciminant variable observable at which the
        `gxs` pdfs are evaluated
    weights : array
        The values of the weights
    Nxs : list of float or tuple of float
        A list of the fitted yield values for the components gxs
    fvals : array_like
        A list of the fitted values of the shape parameters p0,....,pn
    fcov : array
        A covariance matrix of the weighted likelihood fit before the
        correction (this is normally available from the minmiser e.g. iminuit)
    dw_dW_fs : list of callable
        A list of the functions describing the partial derivate of the weight
        function with respect to the W matrix elements (see the tutorial to see
        this passed for sweights and COWs)
    Wvals : list of float or tuple of float
        A list of the W matrix elements
    verbose : bool, optional
        Print some output

    Returns
    -------
    array :
        The corrected covariance matrix

    Notes
    -----
    This function corresponds to the weighted score function with sweights
    (both terms of Eq.51 in
    `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_).
    If the shape parameters of the `gxs` are not known then the full sandwich
    estimate must be used which is not yet implemented in this package.

    """
    dim_kl = len(fvals)
    assert fcov.shape[0] == dim_kl and fcov.shape[1] == dim_kl

    dim_xy = len(gxs)
    assert len(Nxs) == dim_xy
    dim_E = int(
        dim_xy * (dim_xy + 1) / 2
    )  # indepdent elements of symmetric Wxy matrix is n(n+1)/2
    assert len(dw_dW_fs) == dim_E
    assert len(Wvals) == dim_E

    HHpH_term = approx_cov_correct(hs, hdata, weights, fvals, fcov, verbose=False)

    # now construct the E and C' matrices
    Ekl = np.empty((dim_kl, dim_E))
    deriv = point_derivative(hs, fvals, hdata, step=1e-3 * hdata)
    for j in range(dim_kl):
        gxevs = [gx(gdata) for gx in gxs]
        for xy in range(dim_E):
            Ekl[j, xy] = np.sum(dw_dW_fs[xy](*Wvals, *gxevs) * deriv[j])

    Ckl = np.empty((dim_E, dim_E))
    gtot = np.sum([Nxs[i] * gxs[i](gdata) for i in range(dim_xy)], axis=0)

    Citer = [(gxs[i], gxs[j]) for i in range(dim_xy) for j in range(dim_xy) if i >= j]

    for i, gis in enumerate(Citer):
        for j, gjs in enumerate(Citer):
            Ckl[i, j] = np.sum(
                gis[0](gdata) * gis[1](gdata) * gjs[0](gdata) * gjs[1](gdata) / gtot**4
            )

    HECEH_term = fcov @ (Ekl @ Ckl @ Ekl.T) @ fcov.T

    tcov = HHpH_term - HECEH_term

    if verbose:
        print("Full covariance correction for weighted events")
        print("  Original covariance:")
        print("\t", str(fcov).replace("\n", "\n\t "))
        print("  Corrected covariance:")
        print("\t", str(tcov).replace("\n", "\n\t "))

    return tcov


# deprecated alias
approx_cov_correct = covariance_weighted_ml_fit
