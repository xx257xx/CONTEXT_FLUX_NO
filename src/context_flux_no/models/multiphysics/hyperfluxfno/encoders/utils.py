import warnings
from typing import Literal

from jaxtyping import PRNGKeyArray

from .base import AbstractEncoder
from .dpot_encoder import DPOTEncoder
from .trecvit_encoder import TRecViTEncoder
from .vit_encoder import ViTEncoder


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
