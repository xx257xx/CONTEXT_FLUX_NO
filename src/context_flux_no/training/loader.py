import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .dataset import PDEDataset


class SegmentLoader(eqx.Module):
    """A class that loads fixed data trajectories from given dataset."""

    dataset: PDEDataset
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
            u_seg = jax.lax.dynamic_index_in_dim(
                u_pde,
                jax.random.randint(keys[1], (), 0, self.dataset.num_ic),
                keepdims=False,
            )

            n_t = len(self.dataset.t)
            start_ind = jax.random.randint(
                keys[2],
                (),
                0,
                n_t - self.segment_length,
            )
            segment = jax.lax.dynamic_slice_in_dim(
                u_seg, start_ind, self.segment_length
            )

            return segment

        batch = (
            _load_single(jax.random.split(key_batch, self.batch_size)),
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
