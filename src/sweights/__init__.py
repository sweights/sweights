"""Python interface to produce sweights and cows.

* Code: https://github.com/matthewkenzie/sweights
* Docs: https://sweights.readthedocs.io

"""
__version__ = "0.0.4"
__all__ = [
    "sweight",
    "convertRooAbsPdf",
    "cow",
    "kendall_tau",
    "cov_correct",
    "approx_cov_correct",
]

from sweights.sweight import sweight, convertRooAbsPdf
from sweights.cow import cow
from sweights.independence import kendall_tau
from sweights.covariance import cov_correct, approx_cov_correct
