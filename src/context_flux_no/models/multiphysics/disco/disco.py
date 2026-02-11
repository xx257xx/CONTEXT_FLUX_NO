from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from ....nn.misc import destandardize, standardize
from .hypernet import HyperNetwork
from .operatornet import OperatorNetwork


class LinearVariableInFeatures(eqx.Module):
    """Same as equinox.nn.Linear, but accepts arrays with varying in_features."""


class DISCO(eqx.Module):
    """JAX implementation of the DISCO model presented in [1].

    [1] R. Morel et al. DISCO: learning to DISCover an evolution Operator for
    multi-physics-agnostic prediction. ICML (2025)."""

    hypernet: HyperNetwork
    operatornet: OperatorNetwork

    num_spatial_dims: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        embedding_dim: int,
        patch_size: int,
        num_hypernet_blocks: int,
        droppath: int,
        num_hypernet_heads: int,
        mlp_hidden_dim: int,
        boundary_condition: str,
        activation: Callable = jax.nn.gelu,
        rtol: float = 5e-6,
        atol: float = 1e-9,
        max_steps: int = 32,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        # For now, will consider fixed spatial dimensions
        # TODO: Do I need to specify padding mode based on boundary conditions?
        self.hypernet = HyperNetwork(
            in_channels=channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_blocks=num_hypernet_blocks,
            droppath=droppath,
            num_heads=num_hypernet_heads,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )

        self.operatornet = OperatorNetwork(
            num_spatial_dims=num_spatial_dims,
            channels=channels,
            hidden_channels_base=8,
            groups_norm=4,
            boundary_condition=boundary_condition,
            activation=activation,
            dtype=dtype,
            key=keys[1],
        )

        self.mlp_params_common = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=mlp_hidden_dim,
            width_size=mlp_hidden_dim,
            activation=activation,
            final_activation=activation,
            dtype=dtype,
            key=keys[2],
        )

        self.num_spatial_dims = num_spatial_dims
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def time_integrate(self, du_dt, u0, params):
        rhs = dfx.ODETerm(du_dt)
        sol = dfx.diffeqsolve(
            rhs,
            solver=dfx.Bosh3(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=u0,
            args=params,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=self.rtol, atol=self.atol),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=self.max_steps,
        )
        return sol.ys

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " channels *grids"]:
        # Normalize input
        axis_spatial = tuple(range(2, u.ndim))
        u, stats_global = standardize(u, axis=(0,) + axis_spatial)

        u_latent: Float[Array, "time embedding_dim *grids"] = self.hypernetwork(
            u, key=key
        )
        param_latent: Float[Array, " embedding_dim"] = reduce(
            u_latent, "T E ... -> E", "mean"
        )
        param_latent = Float[Array, " mlp_hidden_dim"] = self.mlp_params_common(
            param_latent
        )

        u0, stats = standardize(u[-1], axis=axis_spatial)

        u1 = self.time_integrate(du_dt, u0, params)
        u1: Float[Array, " channels *grids"] = destandardize(u1, **stats)

        u1 = destandardize(jnp.expand_dims(u1, axis=0), **stats_global)
        u1 = jnp.squeeze(u1, axis=0)
        return u1
