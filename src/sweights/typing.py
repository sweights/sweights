"""Types used by the package."""

from typing import Any, Callable, Tuple
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]
Density = Callable[[FloatArray], FloatArray]

Range = Tuple[float, float]

RooAbsPdf = Any
RooRealVar = Any
