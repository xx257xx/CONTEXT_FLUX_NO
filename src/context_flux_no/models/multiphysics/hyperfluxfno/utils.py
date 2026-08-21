import warnings
from typing import Literal

import jax.nn as jnn
from jaxtyping import PRNGKeyArray

from .encoders import AbstractEncoder, DPOTEncoder, TRecViTEncoder, ViTEncoder
from .target_networks import (
    AbstractTargetNetwork,
    FluxNOTargetNetwork,
    FNOTargetNetwork,
    UNetTargetNetwork,
)


def _get_activation(name: str):
    if name.startswith("jax.nn."):
        name = name.removeprefix("jax.nn.")

    if not hasattr(jnn, name):
        raise ValueError(f"Unknown activation function: {name}")

    return getattr(jnn, name)

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
            encoder_kwargs = dict(encoder_kwargs)

            activation = encoder_kwargs.get("activation")

            if isinstance(activation, str):

                encoder_kwargs["activation"] = _get_activation(activation)
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

TARGET_NETWORK_DICT = {
    "UNet": UNetTargetNetwork,
    "FNO": FNOTargetNetwork,
    "FluxNO": FluxNOTargetNetwork,
}


def make_target_network(
    target_network_type: Literal["UNet", "FNO", "FluxNO"],
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    *,
    key: PRNGKeyArray,
    **target_network_kwargs,
) -> AbstractTargetNetwork:
    target_network_fn = TARGET_NETWORK_DICT[target_network_type]
    target_network = target_network_fn(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        key=key,
        **target_network_kwargs,
    )
    return target_network