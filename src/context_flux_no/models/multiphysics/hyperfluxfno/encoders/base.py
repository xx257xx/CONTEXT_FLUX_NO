import abc

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractEncoder(eqx.Module):
    """Abstract base class for encoder modules that take in consecutive time segment of
    shape (time, channels, *spatial_dims) and return an encoded latent vector of shape
    (embedding_dim, ).

    AbstractEncoder subclasses with in_timesteps=None correspond to encoders that can
    accept inputs with varying in_timesteps."""

    num_spatial_dims: eqx.AbstractVar[int]
    in_channels: eqx.AbstractVar[int]
    in_timesteps: eqx.AbstractVar[int | None]
    embedding_dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        u: Float[Array, "time in_channels *spatial_dims"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " embedding_dim"]:
        pass
