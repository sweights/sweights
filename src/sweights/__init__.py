"""Python interface to produce sweights and cows.

* Code: https://github.com/matthewkenzie/sweights
* Docs: https://sweights.readthedocs.io

"""
__version__ = "1.0.0"
__all__ = [
    "SWeight",
    "Cow",
    "convert_rf_pdf",
    "kendall_tau",
    "plot_indep_scatter",
    "cov_correct",
    "approx_cov_correct",
]

from sweights.sweight import SWeight, convert_rf_pdf
from sweights.cow import Cow
from sweights.independence import kendall_tau, plot_indep_scatter
from sweights.covariance import cov_correct, approx_cov_correct
