from sweights.typing import Density


def test_Density():
    def pdf1(x, mu, sigma): ...
    def pdf2(x): ...

    # isinstance does not check signatures :(
    assert isinstance(pdf1, Density)  # should be False
    assert isinstance(pdf2, Density)
