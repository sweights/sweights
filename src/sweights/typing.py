"""Types used by the package."""

from typing import Any, Protocol, Tuple, runtime_checkable
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.floating[Any]]


@runtime_checkable
class Density(Protocol):
    """Density type."""

    def __call__(self, x: FloatArray) -> FloatArray:
        """Return density at x."""
        ...


Range = Tuple[float, float]

RooAbsPdf = Any
RooRealVar = Any
