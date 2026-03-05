from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ..custom_types import IntScalar
from .dataset import PDEDataset


class SegmentLoader(eqx.Module):
    """A class that loads fixed data trajectories from given dataset."""

    dataset: PDEDataset
    segment_length: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    batching_strategy: Literal["random", "consecutive_segments"] = "random"

    def __check_init__(self):
        if self.batching_strategy == "consecutive_segments":
            if len(self.dataset.t) <= self.segment_length + self.batch_size:
                raise ValueError(
                    f"""To use consecutive_segments batching strategy, the batch size 
                    must be less than len(self.dataset.t)-self.segment_length=
                    {len(self.dataset.t) - self.segment_length}. 
                    """
                )

    def init(self, seed: int = 0) -> PRNGKeyArray:
        return jax.random.key(seed)

    def select_segment(
        self, idx_traj: IntScalar, idx_t0: IntScalar
    ) -> Float[Array, "segment_length ..."]:
        """Given scalar integer indices for the trajectory (`idx_traj`) as well as the
        first time point (`idx_t0`), return a trajectory segment of length
        `self.segment_length` starting from time index `idx_t0`."""
        u_traj = jax.lax.dynamic_index_in_dim(
            rearrange(self.dataset.u, "pde ic ... -> (pde ic) ..."),
            index=idx_traj,
            axis=0,
            keepdims=False,
        )
        u_seg = jax.lax.dynamic_slice_in_dim(
            u_traj, start_index=idx_t0, slice_size=self.segment_length, axis=0
        )
        return u_seg

    def load_batch(self, loader_state):
        loader_state_next, key_traj, key_t0 = jax.random.split(loader_state, 3)

        if self.batching_strategy == "random":
            inds_traj = jax.random.randint(
                key_traj,
                (self.batch_size,),
                minval=0,
                maxval=self.dataset.num_trajectories,
            )
            inds_t0 = jax.random.randint(
                key_t0,
                (self.batch_size,),
                minval=0,
                maxval=len(self.dataset.t) - self.segment_length,
            )
            segments = jax.vmap(self.select_segment)(inds_traj, inds_t0)

        elif self.batching_strategy == "consecutive_segments":
            idx_traj = jax.random.randint(
                key_traj,
                (),
                minval=0,
                maxval=self.dataset.num_trajectories,
            )
            idx_t0 = jax.random.randint(
                key_t0,
                (),
                minval=0,
                maxval=len(self.dataset.t) - self.segment_length - self.batch_size,
            )
            inds_t0 = jnp.arange(idx_t0, idx_t0 + self.batch_size, dtype=idx_t0.dtype)
            segments = jax.vmap(self.select_segment, in_axes=(None, 0))(
                idx_traj, inds_t0
            )

        batch = (
            segments,
            self.dataset.dt,
            self.dataset.dx,
        )
        return batch, loader_state_next


class ContextSegmentLoader(eqx.Module):
    """A class that loads fixed length contexts and data trajectories from given
    dataset."""

    dataset: PDEDataset
    context_size: int = eqx.field(static=True)
    segment_length: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def init(self, seed: int = 0) -> PRNGKeyArray:
        return jax.random.key(seed)

    def load_batch(self, loader_state):
        loader_state_next, key_batch = jax.random.split(loader_state, 2)

        @jax.vmap
        def _load_single(key_: PRNGKeyArray):
            keys = jax.random.split(key_, 3)
            u_pde = jax.lax.dynamic_index_in_dim(
                self.dataset.u,
                jax.random.randint(keys[0], (), 0, self.dataset.num_pde),
                keepdims=False,
            )
            u_seg, u_ctx = u_pde.at[
                jax.random.randint(keys[1], (2,), 0, self.dataset.num_ic)
            ].get()

            n_t = len(self.dataset.t)
            start_indices = jax.random.randint(
                keys[2],
                (2,),
                0,
                jnp.array((n_t - self.segment_length, n_t - self.context_size)),
            )
            segment = jax.lax.dynamic_slice_in_dim(
                u_seg, start_indices[0], self.segment_length
            )
            context = jax.lax.dynamic_slice_in_dim(
                u_ctx, start_indices[0], self.context_size
            )
            return context, segment

        batch = (
            *_load_single(jax.random.split(key_batch, self.batch_size)),
            self.dataset.dt,
            self.dataset.dx,
        )
        return batch, loader_state_next
