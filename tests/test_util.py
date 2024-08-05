import pytest
import numpy as np
from numpy.testing import assert_allclose
from sweights import util
from scipy.integrate import quad
from scipy.stats import norm, expon
from typing import Annotated
from iminuit.typing import Interval
import sys
from pathlib import Path


def test_import_optional_module():
    sys.path.append(Path(__file__).parent)
    m = util.import_optional_module("dummy_module", min_version="6")
    assert m.__version__ == "6.30/04"
    with pytest.raises(ImportError):
        util.import_optional_module("dummy_module", min_version="6.31")
    with pytest.raises(ImportError):
        util.import_optional_module("dummy_module", min_version="7")
    with pytest.raises(
        ImportError,
        match=r"dummy_module found, but does not match minimum version 6\.30\.05",
    ):
        util.import_optional_module("dummy_module", min_version="6.30/05")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"npoints": 0},
        {"npoints": 1000},
        {"npoints": 1000, "method": "pchip"},
        {"npoints": 100, "forcenorm": True},
    ),
)
def test_convert_rf_pdf(kwargs):
    R = pytest.importorskip("ROOT")

    mass = R.RooRealVar("m", "m", 0, 3)

    slope = R.RooRealVar("lb", "lb", 0, 2, 1)
    pdf1 = R.RooExponential("bkg", "bkg", mass, slope)
    pdf2 = util.convert_rf_pdf(pdf1, mass, **kwargs)

    x = np.linspace(mass.getMin(), mass.getMax())

    y1 = []
    for xi in x:
        mass.setVal(xi)
        y1.append(pdf1.getVal(mass))

    y2 = pdf2(x)

    assert_allclose(y1, y2, atol=1e-5 if kwargs["npoints"] > 0 else 1e-10)


def test_normalized():
    def fn(x):
        return x + 1

    xrange = np.array((0.0, 1.0))
    fn2 = util.normalized(fn, xrange)

    def integral(x):
        return 0.5 * x**2 + x

    norm = np.diff(integral(xrange))

    x = np.linspace(*xrange)
    assert_allclose(fn2(x), fn(x) / norm)


def test_pdf_from_histogram():
    xe = np.array([0.0, 1.0, 1.5])
    w = np.array([1.0, 3.0])
    fn = util.pdf_from_histogram(w, xe)

    integral = np.sum(np.diff(xe) * np.array([fn(0.0), fn(1.0)]))
    assert_allclose(integral, 1.0)

    x = np.array([-0.1, 0.0, 0.5, 0.99, 1.0, 1.1, 1.49, 1.5, 10.0])
    y = [0.0, 0.25, 0.25, 0.25, 1.5, 1.5, 1.5, 0, 0]
    assert_allclose(fn(x), y)


@pytest.mark.parametrize("order", (1, 2, 3))
def test_make_bernstein_pdf(order):
    range = -1.0, 2.0
    pdfs = util.make_bernstein_pdf(order, *range)
    integrals = [quad(pdf, *range)[0] for pdf in pdfs]
    assert_allclose(integrals, np.ones(order + 1))


def test_make_weighted_negative_log_likelihood():
    mutil = pytest.importorskip("iminuit.util")
    mtyping = pytest.importorskip("iminuit.typing")

    w = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0, 3.0])

    Positive = mtyping.Annotated[float, mtyping.Gt(0)]

    def model(x, a: Positive, b):
        return a + x**b

    ref = -2 * np.sum(w * np.log(model(x, 1, 2)))

    nll = util.make_weighted_negative_log_likelihood(x, w, model)

    assert mutil.describe(nll, annotations=True) == {
        "a": (0, np.inf),
        "b": None,
    }
    assert_allclose(nll(1, 2), ref)


def test_fit_mixture_1():
    rng = np.random.default_rng(1)
    d1 = expon(0, 0.5)
    d2 = norm(0.5, 0.1)
    x1 = d1.rvs(100, random_state=rng)
    x2 = d2.rvs(200, random_state=rng)
    x = np.append(x1, x2)
    yields, list_of_kwargs = util.fit_mixture(x, (d1.pdf, d2.pdf))
    assert_allclose(yields, (100, 200), atol=5)
    assert list_of_kwargs == [{}, {}]


def test_fit_mixture_2():
    rng = np.random.default_rng(1)

    def pdf1(x, slope):
        return expon.pdf(x, 0, slope)

    def pdf2(x, mu: Annotated[float, (0, 1)], sigma: Annotated[float, (1e-3, 2)]):
        return norm.pdf(x, mu, sigma)

    x1 = expon(0, 0.5).rvs(1000, random_state=rng)
    x2 = norm(0.5, 0.1).rvs(2000, random_state=rng)
    x = np.append(x1, x2)

    def model(x, y1, y2, slope, mu, sigma):
        return y1 + y2, y1 * pdf1(x, slope) + y2 * pdf2(x, mu, sigma)

    from iminuit import Minuit
    from iminuit.cost import ExtendedUnbinnedNLL

    c = ExtendedUnbinnedNLL(x, model)
    mi = Minuit(c, 1100, 1800, 0.4, 0.4, 0.2)
    mi.strategy = 0
    mi.migrad()
    assert mi.valid

    yields, list_of_kwargs = util.fit_mixture(
        x,
        (pdf1, pdf2),
        (1100, 1800),
        {pdf1: {"slope": (0.01, 2.0)}},
        {pdf1: {"slope": 0.4}, pdf2: {"mu": 0.4, "sigma": 0.2}},
    )
    assert_allclose(yields, (1000, 2000), atol=10)
    assert [list(kw) for kw in list_of_kwargs] == [["slope"], ["mu", "sigma"]]
    assert_allclose(list_of_kwargs[0]["slope"], 0.5, atol=0.01)
    assert_allclose(list_of_kwargs[1]["mu"], 0.5, atol=0.01)
    assert_allclose(list_of_kwargs[1]["sigma"], 0.1, atol=0.01)


def test_fit_mixture_3():
    # we provoke a fit failure
    rng = np.random.default_rng(1)

    def pdf1(x, slope):
        return expon.pdf(x, 0, slope)

    def pdf2(
        x,
        mu: Annotated[float, Interval(gt=0, lt=1)],
        sigma: Annotated[float, Interval(gt=-1, lt=0)],
    ):
        return norm.pdf(x, mu, sigma)

    x1 = expon(0, 0.5).rvs(1000, random_state=rng)
    x2 = norm(0.5, 0.1).rvs(2000, random_state=rng)
    x = np.append(x1, x2)

    from sweights.util import FitError

    with pytest.raises(FitError):
        util.fit_mixture(x, (pdf1, pdf2), (1100, 1800), {pdf1: {"slope": (0.01, 2.0)}})


def test_GofWarning():
    assert util.GofWarning(0.5).args[0] == "small p-value 0.5 (0.0𝜎), check fit result"
    assert (
        util.GofWarning(norm.sf(1)).args[0]
        == "small p-value 0.16 (1.0𝜎), check fit result"
    )
    assert (
        util.GofWarning(norm.sf(3)).args[0]
        == "small p-value 0.0013 (3.0𝜎), check fit result"
    )
    assert (
        util.GofWarning(norm.sf(5)).args[0]
        == "small p-value 2.9e-07 (5.0𝜎), check fit result"
    )
