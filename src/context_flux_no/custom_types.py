from collections.abc import Callable
from typing import Any, TypeAlias

from jaxtyping import Array, Complex, Float, Int, PyTree


IntScalar:TypeAlias = Int[Array, ""]
FloatScalar:TypeAlias = Float[Array, ""]

FloatArray: TypeAlias = Float[Array, "..."]
ComplexArray: TypeAlias = Complex[Array, "..."]

"""
A filter specification. Typically used on a pytree to filter out certain subtrees. 
Boolean values are treated as-is, while callables are called on each element of the 
pytree. If the callable returns True, the element is kept, otherwise it is filtered out.
"""
FilterSpec: TypeAlias = PyTree[bool | Callable[[Any], bool]]
