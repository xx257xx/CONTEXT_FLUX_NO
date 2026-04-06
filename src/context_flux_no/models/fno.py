from collections.abc import Callable

import equinox as eqx
import jax
from einops import pack, unpack
from jaxtyping import Array, Float, PRNGKeyArray

from ..nn.operators import Fourier
from ..nn.operators.fourier_utils import append_grid_channels


## TODO: Implement FNO2D and higher and FluxFNO, test on Burgers
## TODO: Add padding?
class FNO(eqx.Module, strict=True):
    lift_layer: eqx.nn.MLP
    fourier_layers: tuple[Fourier, ...]
    project_layer: eqx.nn.MLP
    activation: Callable

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    frequency_modes: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    width_lift: int = eqx.field(static=True)
    width_project: int = eqx.field(static=True)
    depth_lift: int = eqx.field(static=True)
    depth_project: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)
    residual_connection: bool = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        lift_dim: int,
        depth: int,
        frequency_modes: int,
        out_channels: int | None = None,
        width_lift: int = 128,
        width_project: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        residual_connection: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, depth + 2)

        self.lift_layer = eqx.nn.MLP(
            in_channels + num_spatial_dims if stack_grid else in_channels,
            lift_dim,
            width_lift,
            depth_lift,
            activation,
            dtype=dtype,
            key=keys[0],
        )
        self.fourier_layers = tuple(
            Fourier(
                num_spatial_dims=num_spatial_dims,
                in_channels=lift_dim,
                out_channels=lift_dim,
                frequency_modes=frequency_modes,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for k in keys[1:-1]
        )

        out_channels = in_channels if out_channels is None else out_channels
        self.project_layer = eqx.nn.MLP(
            lift_dim,
            out_channels,
            width_project,
            depth_project,
            activation,
            dtype=dtype,
            key=keys[-1],
        )

        self.activation = activation
        self.in_channels = in_channels
        self.lift_dim = lift_dim
        self.depth = depth
        self.frequency_modes = frequency_modes
        self.out_channels = out_channels
        self.width_lift = width_lift
        self.width_project = width_project
        self.depth_lift = depth_lift
        self.depth_project = depth_project
        self.stack_grid = stack_grid
        self.residual_connection = residual_connection

    @property
    def layers(self) -> tuple[eqx.Module, ...]:
        return (self.lift_layer, *self.fourier_layers, self.project_layer)

    def __call__(
        self, v: Float[Array, "in_channels grids"]
    ) -> Float[Array, "out_channels grids"]:
        if self.stack_grid:
            v = append_grid_channels(v)

        v = self._apply_channelwise(self.lift_layer, v)
        for fourier in self.fourier_layers:
            v = v + fourier(v) if self.residual_connection else fourier(v)
        v = self._apply_channelwise(self.project_layer, v)
        return v

    def _apply_channelwise(
        self, layer, x: Float[Array, " channels *grids"]
    ) -> Float[Array, " channels_out *grids"]:
        x_, ps = pack([x], "C *")
        y_ = eqx.filter_vmap(layer, in_axes=-1, out_axes=-1)(x_)
        return unpack(y_, ps, "C *")[0]
