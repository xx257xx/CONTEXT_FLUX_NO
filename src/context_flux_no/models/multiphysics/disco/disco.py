from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from ....nn.misc import destandardize, standardize
from ....utils import num_parameters
from ..abstract import AbstractMultiphysicsOperator
from .hypernet import HyperNetwork
from .operatornet import OperatorNetwork


class LinearVariableInFeatures(eqx.Module):
    """Same as equinox.nn.Linear, but accepts arrays with varying in_features."""


class HyperNetworkHead[NN: eqx.Module](eqx.Module):
    linear: eqx.nn.Linear

    target_static: NN
    unflatten_fn: Callable
    where: Callable
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        base_network: NN,
        where,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        target_subnetwork = where(base_network)

        out_size = num_parameters(target_subnetwork)
        self.linear = eqx.nn.Linear(
            in_features=in_size,
            out_features=out_size,
            dtype=dtype,
            key=key,
        )

        target_params, target_static = eqx.partition(
            target_subnetwork, eqx.is_inexact_array
        )
        _, unflatten_fn = jax.flatten_util.ravel_pytree(target_params)
        self.target_static: NN = target_static
        self.unflatten_fn = unflatten_fn
        self.where = where
        self.in_size = in_size
        self.out_size = out_size

    def param_vector_to_target(self, params: Float[Array, " out_size"]) -> NN:
        target_params = self.unflatten_fn(params)
        target = eqx.combine(target_params, self.target_static)
        return target

    def __call__(
        self,
        input_: Float[Array, " in_size"],
        base_network: NN,
        target_transform: Callable[[NN], NN] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NN:
        params_flat = self.linear(input_)
        target = self.param_vector_to_target(params_flat)
        if target_transform is not None:
            target = target_transform(target)
        return eqx.tree_at(self.where, base_network, target)


class DISCO(AbstractMultiphysicsOperator):
    """JAX implementation of the DISCO model presented in [1].

    [1] R. Morel et al. DISCO: learning to DISCover an evolution Operator for
    multi-physics-agnostic prediction. ICML (2025)."""

    hypernet: HyperNetwork
    operatornet: OperatorNetwork
    mlp_params_common: eqx.nn.MLP
    mlp_params_heads: list[HyperNetworkHead]
    operatornet_scale: PyTree

    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
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
            depth=1,
            activation=activation,
            final_activation=activation,
            dtype=dtype,
            key=keys[2],
        )

        wheres = [
            lambda net: (net.input_block.layers[0], net.output_block.layers[0]),
            lambda net: (
                *net.input_block.layers[1:],
                *net.output_block.layers[1:],
                net.downsampling_blocks,
                net.upsampling_blocks,
            ),
            lambda net: net.bottleneck,
        ]
        keys_h = jax.random.split(keys[3], len(wheres))
        self.mlp_params_heads = [
            HyperNetworkHead(mlp_hidden_dim, self.operatornet, w, dtype, key=k)
            for (w, k) in zip(wheres, keys_h)
        ]

        # Pre-compute per-weight scale values
        self.operatornet_scale = jax.tree.map(
            lambda x: float(jnp.max(jnp.abs(x))),
            eqx.filter(self.operatornet, eqx.is_inexact_array),
        )

        self.num_spatial_dims = num_spatial_dims
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def scale_operatornet(self, operatornet: OperatorNetwork) -> OperatorNetwork:
        params, static = eqx.partition(operatornet, eqx.is_inexact_array)

        def scale_fn(weight: Array, scale: float, factor: float = 2.0):
            scale_ = scale * factor
            return scale_ * (2 * jax.nn.sigmoid(weight / scale_) - 1)

        params_scaled = jax.tree.map(
            scale_fn, params, jax.lax.stop_gradient(self.operatornet_scale)
        )
        return eqx.combine(params_scaled, static)

    def time_integrate(self, du_dt, u0):
        rhs = dfx.ODETerm(du_dt)
        sol = dfx.diffeqsolve(
            rhs,
            solver=dfx.Bosh3(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=u0,
            args=None,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(
                rtol=self.rtol,
                atol=self.atol,
                dtmin=1.0 / self.max_steps,
                force_dtmin=True,
            ),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=2 * self.max_steps,
        )
        return sol.ys[0]

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " channels *grids"], None]:
        # Normalize input
        axis_spatial = tuple(range(2, u.ndim))
        u, stats_global = standardize(u, axis=(0,) + axis_spatial)

        u_latent: Float[Array, "time embedding_dim *grids"] = self.hypernet(
            u, key=key, inference=inference
        )
        param_latent: Float[Array, " embedding_dim"] = reduce(
            u_latent, "T E ... -> E", "mean"
        )
        param_latent: Float[Array, " mlp_hidden_dim"] = self.mlp_params_common(
            param_latent
        )

        opnet = self.operatornet

        # Update operatornet with hypernetwork generated weights
        for param_head in self.mlp_params_heads:
            opnet = param_head(param_latent, opnet)

        opnet = self.scale_operatornet(opnet)

        u0 = u[-1]
        u0, stats = standardize(u0, axis=tuple(range(1, u0.ndim)))
        u1 = self.time_integrate(lambda t, u_, args: opnet(u_), u0)

        u1: Float[Array, " channels *grids"] = destandardize(u1, **stats)
        u1 = destandardize(jnp.expand_dims(u1, axis=0), **stats_global)
        u1 = jnp.squeeze(u1, axis=0)
        return u1, None
