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
