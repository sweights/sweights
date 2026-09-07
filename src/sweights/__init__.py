"""Python interface to produce sweights and cows."""

__all__ = [
    "Cow",
    "Cows",
    "SWeight",
    "__version__",
    "approx_cov_correct",
    "convert_rf_pdf",
    "cov_correct",
    "kendall_tau",
    "plot_indep_scatter",
]

from importlib.metadata import version

from .covariance import approx_cov_correct, cov_correct
from .cow import Cow
from .experimental import Cows
from .independence import kendall_tau, plot_indep_scatter
from .sweight import SWeight
from .util import convert_rf_pdf

__version__ = version("sweights")
