"""Various utilities used by the package or in the tutorials."""

from packaging.version import Version
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.integrate import quad
from scipy.special import comb
from scipy.stats import chi2, norm
import warnings
from typing import (
    Tuple,
    Optional,
    Union,
    Any,
    TYPE_CHECKING,
    List,
    Callable,
    Sequence,
    Dict,
)
from .typing import RooAbsPdf, RooRealVar, Density, FloatArray, Range
from numpy.typing import ArrayLike
from iminuit import Minuit
from iminuit.util import describe
from iminuit.cost import ExtendedUnbinnedNLL
import enum

__all__ = [
    "convert_rf_pdf",
    "plot_binned",
    "normalized",
    "pdf_from_histogram",
    "BernsteinBasisPdf",
    "make_bernstein_pdf",
    "make_norm_pdf",
    "make_weighted_negative_log_likelihood",
]


def import_optional_module(name: str, *, min_version: str = "") -> Any:
    """
    Import an optional dependency.

    Users are not supposed to call this themselves. In this package, we use optional
    dependencies in some places. We enhance the standard error message to make it more
    helpful and optionally check whether the version of the package matches the required
    minimum.

    """
    from importlib import import_module

    try:
        mod = import_module(name)

        if min_version:
            version = getattr(mod, "__version__", "0")
            version = _normalize_version(version)
            min_version = _normalize_version(min_version)

            if not Version(min_version) <= Version(version):
                msg = f"{name} found, but does not match minimum version {min_version}"
                raise ImportError(msg)

        return mod

    except ModuleNotFoundError as e:
        e.msg += (
            " This is an optional dependency, "
            "please install it manually to use this function."
        )
        raise


def _normalize_version(version: str) -> str:
    """Replace / with . in non-standard ROOT version string."""
    return version.replace("/", ".")


def convert_rf_pdf(
    pdf: RooAbsPdf,
    obs: RooRealVar,
    *,
    npoints: int = 0,
    method: str = "makima",
    forcenorm: bool = True,
) -> Density:
    """
    Convert a RooFit RooAbsPdf into a vectorized Python callable.

    This converts a RooFit::RooAbsPdf object into a Python callable that can be used by
    either the :class:`SWeight` or :class:`Cow` classes.

    Parameters
    ----------
    pdf : RooAbsPdf
        The pdf, must inherit from RooAbsPdf (e.g. RooGaussian, RooExponential,
        RooAddPdf etc.)
    obs : RooRealVar
        The observable.
    npoints: int, optional (default is 0)
        If npoints is zero, a wrapper around the RooFit PDF is returned. This wrapper
        internally calls the RooFit PDF and therefore produces exact results. If npoints
        is larger than zero, a spline interpolator is constructed instead to approximate
        the RooFit PDF. It is constructed by evaluating the exact PDF at the given
        number of points, which are equally spaced over the range of ``obs``, which must
        be a bounded variable.  The spline is an approximation, but faster to compute.
    method: str, optional (default is "makima")
        Interpolation method to use. Accepted values are "makima" and "pchip".
    forcenorm : bool, optional (default is True)
        This only has an effect, if an interpolator is returned. Since the interpolator
        is not normalized in general, we need to normalize it. Setting this to False
        skips the computation of the integral, which is done numerically. Deactivate
        this only if the numerical interaction fails for some reason.

    Returns
    -------
    callable :
        A callable function representing a normalised pdf which can then be
        passed to the :class:`SWeight` or :class:`Cow` classes

    """
    R = import_optional_module("ROOT", min_version="6")

    assert isinstance(obs, R.RooAbsReal)

    range = (obs.getMin(), obs.getMax())

    if npoints > 0:
        x = np.linspace(*range, npoints)
        y = []

        for xi in x:
            obs.setVal(xi)
            y.append(pdf.getVal([obs]))

        # We only allow makima and pchip interpolators, because these do not overshoot.
        # This guarantees that the interpolators do not return negative values.
        fn: Density
        if method == "makima":
            fn = Akima1DInterpolator(x, y, method="makima")
        elif method == "pchip":
            fn = PchipInterpolator(x, y)
        else:
            msg = (
                f"method='{method}' is not recognized, "
                "allowed values are 'makima' and 'pchip'"
            )
            raise ValueError(msg)

        if forcenorm:
            with warnings.catch_warnings():
                # ignore accuracy warnings from integration
                warnings.simplefilter("ignore")
                return normalized(fn, range)

    else:
        wrapper = getattr(R, "RooAbsPdfPythonWrapper", None)
        if wrapper is None:
            R.gInterpreter.Declare(
                """std::vector<double> RooAbsPdfPythonWrapper(
                    const std::vector<double>& x, RooAbsPdf* pdf, RooRealVar* obs) {{
            std::vector<double> result;
            result.reserve(x.size());
            RooArgSet nset(*obs);
            for (const auto& xi : x) {{
                obs->setVal(xi);
                result.push_back(pdf->getVal(nset));
            }}
            return result;
}}"""
            )
            wrapper = getattr(R, "RooAbsPdfPythonWrapper")

        def fn(x: FloatArray) -> FloatArray:
            r = wrapper(x, pdf, obs)
            return np.array(r)

    return fn


def plot_binned(
    data: ArrayLike,
    *,
    bins: Optional[Union[int, FloatArray]] = 100,
    range: Optional[Range] = None,
    weights: Optional[ArrayLike] = None,
    density: bool = False,
    **kwargs: Any,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Plot histogram from data.

    Parameters
    ----------
    data: array-like
        Data to sort into a histogram.
    bins: int or array-like or None, optional
        Number of bins of the histogram.
    range: (float, float) or None, optional
        Range of the histogram.
    weights: array-like or None, optional
        Weights.
    density: bool, optional (default is False)
        If True, normalize the histogram.
    axes: Axes or None, optional
        Axes to plot on. If None, then use matplotlib.pyplot.gca().
    **kwargs:
        Further arguments are forwarded to matplotlib.pyplot.errorbar.

    """
    if TYPE_CHECKING:
        # workaround for buggy histogram annotations
        assert bins is not None
    if weights is None:
        val, xe = np.histogram(data, bins=bins, range=range)
        err = val**0.5
    else:
        wsum, xe = np.histogram(data, bins=bins, range=range, weights=weights)
        w2sum = np.histogram(data, bins=xe, weights=np.asarray(weights) ** 2)[0]
        val = wsum
        err = np.sqrt(w2sum)
    cx = 0.5 * (xe[1:] + xe[:-1])
    if "marker" in kwargs:
        kwargs["fmt"] = kwargs.pop("marker")
    elif "fmt" not in kwargs:
        kwargs["fmt"] = "o"
    if density:
        f = 1 / (np.sum(val) * np.diff(xe))
        val = val * f
        err = err * f
    plt = import_optional_module("matplotlib.pyplot")
    plt.errorbar(cx, val, err, **kwargs)
    return val, err, xe


def normalized(fn: Density, range: Range) -> Density:
    """Return a function normalized over the given range."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fnorm = quad(fn, *range)[0]
    return lambda x: fn(x) / fnorm


def pdf_from_histogram(w: FloatArray, xe: FloatArray) -> Density:
    """
    Return a pdf (piecewise constant) constructe from a histogram.

    Parameters
    ----------
    w: array of float or int
        Counts of the histogram.
    xe: array of float
        Edges of the histogram.

    Returns
    -------
    Callable
        A normalized density.

    """
    w = w / np.sum(w)
    w /= np.diff(xe)

    def fn(x: FloatArray) -> FloatArray:
        r = np.zeros_like(x)
        ind = np.searchsorted(xe, x, side="right") - 1
        mask = (0 <= ind) & (ind < len(xe) - 1)
        r[mask] = w[ind[mask]]
        return r

    return fn


class BernsteinBasisPdf:
    """Bernstein basis PDF."""

    def __init__(self, k: int, n: int, a: float, b: float):
        """
        Initialize Bernstein basis.

        Parameters
        ----------
        k: int
            Index of the basis.
        n: int
            Order of the polynom.
        a: float
            Starting value where the polynomial is defined and normalized.
        b: float
            Ending value where the polynomial is defined and normalized.

        """
        self._a = a
        self._iw = 1.0 / (b - a)
        self._fnorm: float = comb(n, k) * self._iw * (n + 1)
        self._k = k
        self._ak = n - k

    def __call__(self, x: FloatArray) -> FloatArray:
        """Compute probability density."""
        z: FloatArray = (x - self._a) * self._iw
        az: FloatArray = 1.0 - z
        return z**self._k * az**self._ak * self._fnorm


def make_bernstein_pdf(degree: int, a: float, b: float) -> List[Density]:
    """
    Construct a normalized Bernstein basis of given degree.

    The Bernstein basis polynomials returned by this function
    are normalized over the given range.

    Parameters
    ----------
    degree: int
        Order of the Bernstein basis. Must be at least 1.
    a: float
        Starting value where the polynomial is defined and normalized.
    b: float
        Ending value where the polynomial is defined and normalized.

    Returns
    -------
    Callable
        Normalized density.

    """
    if degree < 1:
        raise ValueError("minimum order is 1")

    return [BernsteinBasisPdf(i, degree, a, b) for i in range(degree + 1)]


class truncnorm:
    def __init__(self, a: float, b: float, mu: float, sigma: float):
        d = norm(mu, sigma)
        self.d = d
        self.norm = d.cdf(b) - d.cdf(a)

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.d.pdf(x) / self.norm  # type:ignore


def make_norm_pdf(
    a: float, b: float, mu: Sequence[float], sigma: float
) -> List[Density]:
    """
    Construct a sequence of truncated normal distributions.

    Parameters
    ----------
    a: float
        Lower end of the data window.
    b: float
        Upper end of the data window.
    mu: array-like
        Locations of the truncated normal distributions.
    sigma: float
        Width of the truncated normal distributions.

    Returns
    -------
    list of truncated normal distributions

    """
    mu, sigma = np.broadcast_arrays(mu, sigma)
    mu, sigma = np.atleast_1d(mu, sigma)
    return [truncnorm(a, b, mui, sigmai) for (mui, sigmai) in zip(mu, sigma)]


def make_weighted_negative_log_likelihood(
    x: FloatArray,
    weights: FloatArray,
    model: Callable[..., FloatArray],
) -> Callable[..., float]:
    """Construct weighted log-likelihood function compatible with iminuit."""
    parameters = {}
    first = True
    for par, limits in describe(model, annotations=True).items():
        if first:
            first = False
            continue
        parameters[par] = limits

    def nll(*args: float) -> float:
        lp = safe_log(model(x, *args))
        return -2 * np.sum(weights * lp)

    nll.errordef = 1.0  # type:ignore
    nll._parameters = parameters  # type:ignore

    return nll


class FitValidation(enum.Enum):
    """How to validate the fit in :func:`fit_mixture`."""

    NONE = 0
    GOF = 1
    DISPLAY = 2
    PLOT = 3


class FitError(RuntimeError):
    pass


def fit_mixture(
    x: FloatArray,
    pdfs: Sequence[Density],
    yields: Optional[Sequence[float]] = None,
    bounds: Dict[Density, Dict[str, Range]] = {},
    starts: Dict[Density, Dict[str, float]] = {},
    validation: FitValidation = FitValidation.GOF,
) -> Tuple[List[float], List[Dict[str, float]]]:
    pdfs = list(pdfs)
    if yields is not None:
        yields = list(yields)
    parameters = [_get_pdf_parameters(pdf) for pdf in pdfs]
    pdf_bounds = []
    pdf_starts = []
    par_names = []
    if any(parameters):
        for i, (pdf, pars) in enumerate(zip(pdfs, parameters)):
            par_names.append(f"yield[{i}]")
            par_names += list(pars)
            bounds_dict = bounds.get(pdf, {})
            starts_dict = starts.get(pdf, {})
            # bounds argument has precedence over bounds from annotations
            bounds_list = [bounds_dict.get(k, pars[k]) for k in pars]
            pdf_bounds.append(bounds_list)
            starts_list = [
                starts_dict.get(k, _guess_starting_value(a, b))
                for k, (a, b) in zip(pars, bounds_list)
            ]
            pdf_starts.append(starts_list)
    else:
        par_names = [f"yield[{i}]" for i in range(len(pdfs))]
    yields, *list_of_vals = _fit_mixture(
        x,
        pdfs,
        yields,
        pdf_bounds,
        pdf_starts,
        par_names,
        validation,
    )
    if list_of_vals:
        list_of_kwargs = [
            {k: v for k, v in zip(pars, vals)}
            for pars, vals in zip(parameters, list_of_vals)
        ]
    else:
        list_of_kwargs = [{}] * len(yields)
    return yields, list_of_kwargs


def _make_model_from_parametric_pdfs(
    pdfs: Sequence[Density], slices: List[Any]
) -> Callable[..., Tuple[float, FloatArray]]:

    def model(x: FloatArray, *par: float) -> Tuple[float, FloatArray]:
        yields, *pars = (par[sl] for sl in slices)
        fint = 0.0
        f = np.zeros_like(x)
        for y, pdf, par in zip(yields, pdfs, pars):
            f += y * pdf(x, *par)
            fint += y
        return fint, f

    return model


def _make_model_from_fixed_pdfs(
    pdfs: Sequence[Density],
) -> Callable[..., Tuple[float, FloatArray]]:
    def model(x: FloatArray, *par: float) -> Tuple[float, FloatArray]:
        fint = 0.0
        f = np.zeros_like(x)
        for y, pdf in zip(par, pdfs):
            f += y * pdf(x)
            fint += y
        return fint, f

    return model


def _fit_mixture(
    x: FloatArray,
    pdfs: List[Density],
    yield_starts: Optional[List[float]],
    bounds: List[List[Range]],
    starts: List[List[float]],
    names: List[str],
    validation: FitValidation,
) -> List[List[float]]:
    assert len(bounds) == len(starts)
    slices = [slice(0, len(pdfs))]
    ipar = len(pdfs)
    for s in starts:
        slices.append(slice(ipar, ipar + len(s)))
        ipar += len(s)

    if len(slices) > 1:
        model = _make_model_from_parametric_pdfs(pdfs, slices)
    else:
        model = _make_model_from_fixed_pdfs(pdfs)

    if yield_starts is None:
        yield_starts = _guess_starting_yields(len(x), len(pdfs))
    yield_bounds = [(0, np.inf) for _ in range(len(pdfs))]

    starts2: FloatArray = np.concatenate([yield_starts] + starts)
    bounds2: FloatArray = np.concatenate(
        [yield_bounds] + bounds, axis=0  # type:ignore
    )
    nll = ExtendedUnbinnedNLL(x, model)
    min = Minuit(nll, *starts2, name=names)
    min.strategy = 0
    min.limits = bounds2
    min.migrad()
    if validation is FitValidation.DISPLAY:
        try:
            from IPython.display import display

            display(min)
        except ModuleNotFoundError:
            print(min)
    elif validation is FitValidation.PLOT:
        min.visualize()
    elif validation is FitValidation.GOF:

        def pdf(x: FloatArray) -> FloatArray:
            fint, f = model(x, *min.values)
            return f / fint

        pgof = gof_pvalue(x, pdf, min.nfit)
        if pgof < 0.01:
            warnings.warn(GofWarning(pgof), stacklevel=2)
    if not min.valid:
        msgs = ["fit failed", f"{min.fmin}", f"{min.params}"]
        raise FitError("\n".join(msgs))
    return [min.values[sl] for sl in slices]


def _guess_starting_value(a: float, b: float) -> float:
    if a > -np.inf and b < np.inf:
        return 0.5 * (a + b)
    return np.clip(0, a * 1.1 + 1, b * 0.9 - 1)  # type:ignore


def _guess_starting_yields(ndata: int, nyields: int) -> List[float]:
    return [ndata / nyields] * nyields


TINY_FLOAT = np.finfo(float).tiny


def safe_log(x: FloatArray) -> FloatArray:
    # guard against x = 0
    return np.log(np.maximum(TINY_FLOAT, x))  # type:ignore


class GofWarning(UserWarning):
    """Warning emitted if the goodness-of-fit test fails or cannot be carried out."""

    def __init__(self, pvalue: float):
        sigma = norm().isf(pvalue)
        msg = f"small p-value {pvalue:.2g} ({sigma:.1f}𝜎), " "check fit result"
        super().__init__(msg)


def gof_pvalue(
    x: FloatArray, pdf: Density, nfit: int, *, bins: Union[int, ArrayLike] = 0
) -> float:
    ntot = len(x)
    if isinstance(bins, int):
        if bins == 0:
            bins = min(100, ntot // 10)
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    else:
        xmin = x.min()
        xmax = x.max()
        bins = np.atleast_1d(bins)
        if len(bins) < 2:
            raise ValueError("bins array must at least have two entries")
        if bins[0] > xmin or bins[-1] < xmax:
            msg = (
                f"edge range ({bins[0]}, {bins[-1]}) must not be "
                f"narrower than data range ({xmin}, {xmax})"
            )
            raise ValueError(msg)
        edges = np.array(bins).astype(np.float64)
        bins = len(bins)

    if bins < 2:
        warnings.warn("not enough bins to perform test", GofWarning)
        return np.nan

    counts = np.histogram(x, bins=edges)[0]

    # Compute integral over pdf with Simpson's rule to exploit
    # vectorization, but fall back to numerical integration if
    # result differs too much from simple mid-point integration.
    # "Too much" in this context is taken to be more than 1e-3
    # relative deviation.
    pe = pdf(edges)
    pa = pe[:-1]
    pb = pe[1:]
    dx = np.diff(edges)
    pm = pdf(edges[:-1] + 0.5 * dx)
    pn = dx / 6 * (pa + 4 * pm + pb)
    mask = np.abs(pn - dx * pm) > 1e-3 * pn
    for i in np.arange(bins)[mask]:
        pn[i] = _quad_workaround(pdf, *edges[i : i + 2])

    # G-test, test statistic is asymptotically chi-square distributed
    g = np.sum(2 * counts * np.log(counts / (pn * ntot)))
    return chi2(bins - nfit).sf(g)  # type:ignore


def _quad_workaround(
    fn: Callable[[FloatArray], FloatArray], a: float, b: float
) -> float:
    def wrapped(x: float) -> float:
        return fn(np.atleast_1d(x))[0]  # type:ignore

    return quad(wrapped, a, b)[0]  # type:ignore


def _get_pdf_parameters(fn: Density) -> Dict[str, Range]:
    """
    Return PDF paramters as dict with limits.

    The first parameter is skipped, which is the observation.
    """
    result = describe(fn, annotations=True)
    items = iter(result.items())
    next(items)  # skip first entry
    return {
        k: (
            (-np.inf, np.inf)
            if lim is None
            else (
                -np.inf if lim[0] is None else lim[0],
                np.inf if lim[1] is None else lim[1],
            )
        )
        for (k, lim) in items
    }
