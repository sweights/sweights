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
from .sweight import SWeight
from .cow import Cow
from .independence import kendall_tau, plot_indep_scatter
from .covariance import cov_correct, approx_cov_correct
from .util import convert_rf_pdf

__version__ = version("sweights")
