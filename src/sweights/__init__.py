"""Python interface to produce sweights and cows."""

__all__ = [
    "SWeight",
    "Cow",
    "convert_rf_pdf",
    "kendall_tau",
    "plot_indep_scatter",
    "cov_correct",
    "approx_cov_correct",
    "__version__",
]

from importlib.metadata import version
from sweights.sweight import SWeight, convert_rf_pdf
from sweights.cow import Cow
from sweights.independence import kendall_tau, plot_indep_scatter
from sweights.covariance import cov_correct, approx_cov_correct

__version__ = version("sweights")
