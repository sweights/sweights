import pytest
import numpy as np
from numpy.testing import assert_allclose
from sweights import util


def test_convert_rf_pdf():
    R = pytest.importorskip("ROOT")

    mass = R.RooRealVar("m", "m", 0, 3)

    slope = R.RooRealVar("lb", "lb", 0, 2, 1)
    pdf1 = R.RooExponential("bkg", "bkg", mass, slope)
    pdf2 = util.convert_rf_pdf(pdf1, mass)

    x = np.linspace(mass.getMin(), mass.getMax())

    y1 = []
    for xi in x:
        mass.setVal(xi)
        y1.append(pdf1.getVal(mass))

    y2 = pdf2(x)

    assert_allclose(y1, y2)
