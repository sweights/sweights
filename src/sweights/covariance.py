"""Implementation of covariance correction for weighted fits."""

import numpy as np
from scipy.misc import derivative

# derivative of function pdf with respect to variable at index var
# evaluated at point point


def _partial_derivative(pdf, var, point, data):
    args = point[:]

    def wraps(x):
        args[var] = x
        return pdf(data, *args)

    return derivative(wraps, point[var], dx=1e-6)


def approx_cov_correct(pdf, data, wts, fvals, fcov, verbose=False):
    """
    Perform a first order covariance correction for a fit to weighted data.

    Parameters
    ----------
    pdf : callable
        The control variable pdf which must take arguments (x, p0,...,pn)
        which has been fitted to a weighted dataset, where x is the observable
        and p0 ... pn are the shape parameters
    data : ndarray
        The data values of the observable at which the pdf is evaluated
    wts : ndarray
        The values of the weights
    fvals : array_like
        A list of the fitted values of the shape parameters p0,....,pn
    fcov : ndarray
        A covariance matrix of the weighted likelihood fit before the
        correction (this is normally available from the minmiser e.g. iminuit)
    verbose : bool, optional
        Print some output

    Returns
    -------
    ndarray :
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
    dim = len(fvals)
    assert fcov.shape[0] == dim and fcov.shape[1] == dim

    Djk = np.zeros(fcov.shape)

    prob = pdf(data, *fvals)

    for j in range(dim):
        derivj = _partial_derivative(pdf, j, fvals, data)
        for k in range(dim):
            derivk = _partial_derivative(pdf, k, fvals, data)

            Djk[j, k] = np.sum(wts ** 2 * (derivj * derivk) / prob ** 2)

    corr_cov = fcov * Djk * fcov.T

    if verbose:
        print("First order covariance correction for weighted events")
        print("  Original covariance:")
        print("\t", str(fcov).replace("\n", "\n\t "))
        print("  Corrected covariance:")
        print("\t", str(corr_cov).replace("\n", "\n\t "))

    return corr_cov


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
    hdata : ndarray
        The data values of the control variable observable at which the `hs`
        pdf is evaluated
    gdata : ndarray
        The data values of the disciminant variable observable at which the
        `gxs` pdfs are evaluated
    weights : ndarray
        The values of the weights
    Nxs : list of float or tuple of float
        A list of the fitted yield values for the components gxs
    fvals : array_like
        A list of the fitted values of the shape parameters p0,....,pn
    fcov : ndarray
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
    ndarray :
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
    for j in range(dim_kl):
        derivj = _partial_derivative(hs, j, fvals, hdata)
        gxevs = [gx(gdata) for gx in gxs]
        for xy in range(dim_E):
            Ekl[j, xy] = np.sum(dw_dW_fs[xy](*Wvals, *gxevs) * derivj)

    Ckl = np.empty((dim_E, dim_E))
    gtot = np.sum([Nxs[i] * gxs[i](gdata) for i in range(dim_xy)], axis=0)

    Citer = [(gxs[i], gxs[j]) for i in range(dim_xy) for j in range(dim_xy) if i >= j]

    for i, gis in enumerate(Citer):
        for j, gjs in enumerate(Citer):
            Ckl[i, j] = np.sum(
                gis[0](gdata)
                * gis[1](gdata)
                * gjs[0](gdata)
                * gjs[1](gdata)
                / gtot ** 4
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
