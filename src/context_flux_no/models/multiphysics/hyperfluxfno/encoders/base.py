import abc
import warnings
from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.multiphysics.hyperfluxfno.encoders import TRecViTEncoder

from .dpot_encoder import DPOTEncoder
from .vit_encoder import ViTEncoder


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


def make_encoder(
    encoder_type: Literal["ViT", "DPOT", "TRecViT"],
    num_spatial_dims: int,
    in_channels: int,
    embedding_dim: int,
    in_timesteps: int | None = None,
    *,
    key: PRNGKeyArray,
    **encoder_kwargs,
) -> AbstractEncoder:
    match encoder_type:
        case "ViT":
            if num_spatial_dims != 1:
                raise ValueError("ViTEncoder is only supported for num_spatial_dims=1")
            if in_timesteps is not None:
                warnings.warn(
                    """ViTEncoder supports variable in_timesteps. The given in_timesteps
                     value will be ignored."""
                )
            encoder = ViTEncoder(
                in_channels=in_channels,
                embedding_dim=embedding_dim,
                **encoder_kwargs,
                key=key,
            )
        case "DPOT":
            if in_timesteps is None:
                raise ValueError("DPOTEncoder does not support variable in_timesteps.")
            encoder = DPOTEncoder(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                in_timesteps=in_timesteps,
                embedding_dim=embedding_dim,
                **encoder_kwargs,
                key=key,
            )
        case "TRecViT":
            if in_timesteps is not None:
                warnings.warn(
                    """TRecViTEncoder supports variable in_timesteps. The given 
                    in_timesteps value will be ignored."""
                )
            encoder = TRecViTEncoder(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                embedding_dim=embedding_dim,
                **encoder_kwargs,
                key=key,
            )
        case _:
            raise NotImplementedError("Unrecognized encoder type.")

    return encoder