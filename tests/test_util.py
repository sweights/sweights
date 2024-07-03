import pytest
import numpy as np
from numpy.testing import assert_allclose
from sweights import util


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
