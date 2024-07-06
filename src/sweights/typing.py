"""Types used by the package."""

from typing import Any, Protocol, Tuple, runtime_checkable, Callable
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.floating[Any]]


Cost = Callable[..., np.float64]


@runtime_checkable
class Density(Protocol):
    """Type that matches densities."""

    def __call__(self, x: FloatArray) -> FloatArray:
        """Return density at x."""
        ...


## isinstance cannot distinguish between ParametricDensity and Density
## because the signature is not checked. Perhaps this can be fixed in the future
## so we keep this protocol around for now.
#
# @runtime_checkable
# class ParametricDensity(Protocol):
#     """Type that matches densities which have parameters."""

#     def __call__(self, x: FloatArray, arg: float, *args: float) -> FloatArray:
#         """Return density at x."""
#         ...

# AnyDensity = Union[Density, ParametricDensity]


Range = Tuple[float, float]

RooAbsPdf = Any
RooRealVar = Any
