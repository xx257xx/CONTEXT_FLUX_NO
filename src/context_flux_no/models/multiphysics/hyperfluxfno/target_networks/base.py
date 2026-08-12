import abc

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractTargetNetwork(eqx.Module):
    """Abstract base class for target networks that take in the PDE solution field at 
    time t of shape (in_channels, *spatial_dims) as well a tuple containing temporal and 
    spatial grid sizes (dt, *dxs) and return the PDE solution field at the next time 
    t+dt of shape (out_channels, *spatial_dims).
    """

    num_spatial_dims: eqx.AbstractVar[int]
    in_channels: eqx.AbstractVar[int]
    out_channels: eqx.AbstractVar[int]


    @abc.abstractmethod
    def __call__(
        self,
        u: Float[Array, " in_channels *spatial_dims"],
        args: tuple[float, ...],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        pass
