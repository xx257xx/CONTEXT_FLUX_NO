import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

from .batching import AbstractBatching
from .dataset import PDEDataset


class SegmentLoader(eqx.Module):
    """
    Basic implementation of a SegmentLoader, which is a dedicated class that samples a
    given dataset of trajectories  and returns a batch of
    trajectory segments with fixed length.

    This class inspired by DataLoader classes in many deep learning libraries, such as
    `torch.utils.data.DataLoader`.
    """

    dataset: PDEDataset
    segment_length: int
    batch_strategy: AbstractBatching

    def __check_init__(self):
        if self.segment_length < 2:
            raise ValueError("Minimum allowed segment length is 2.")

    @property
    def num_segments_per_traj(self) -> int:
        return len(self.dataset) - self.segment_length + 1

    @property
    def num_total_segments(self) -> int:
        return len(self.dataset) * self.num_segments_per_traj

    @property
    def batch_size(self) -> int:
        batch_size = self.batch_strategy.batch_size
        if batch_size is None:
            return self.num_total_segments
        else:
            return batch_size

    @property
    def num_batches(self) -> int | None:
        """Number of batches required to cover (on average) the entire dataset.

        This number is always rounded up to the nearest integer."""
        return math.ceil(self.num_total_segments / self.batch_size)

    def get_segments(
        self, traj_idx: Int[Array, " batch_size"], time_idx: Int[Array, " batch_size"]
    ) -> tuple[
        Float[Array, "batch_size segment_len dim x"],
        Float[Array, "batch_size segment_len"],
        Float[Array, "batch_size x"],
        Float[Array, "batch_size params"],
    ]:
        @eqx.filter_vmap
        def _get_segments(sample_idx, time_idx):
            t_segment = jax.lax.dynamic_slice_in_dim(
                self.dataset.t, time_idx, self.segment_length
            )
            coeff_segment = jax.lax.dynamic_index_in_dim(
                self.dataset.coeffs, sample_idx, keepdims=False
            )

            u_segment = jax.lax.dynamic_index_in_dim(
                self.dataset.u, sample_idx, keepdims=False
            )
            u_segment = jax.lax.dynamic_slice_in_dim(
                u_segment, time_idx, self.segment_length
            )
            return (
                u_segment,
                t_segment,
                self.dataset.x,
                coeff_segment,
            )

        return _get_segments(traj_idx, time_idx)

    def linear_to_sample_indices(
        self, linear_indices: Int[Array, " {self.batch_size}"]
    ) -> tuple[Int[Array, " {self.batch_size}"], Int[Array, " {self.batch_size}"]]:
        """
        Converts the 1D array of linear indices representing the starting position of
        the segments in the batch to a tuple of indices that can be used to locate the
        said position in `self.dataset.u`.
        """
        return jnp.divmod(linear_indices, self.num_segments_per_traj)

    def init(self) -> PyTree:
        """
        Returns the initial loader_state to be fed into the first call of
        `self.load_batch`.

        This is inspired by optax's optimizer.init function.
        """
        batch_state_init = self.batch_strategy.init(self.num_total_segments)
        return (batch_state_init,)

    def load_batch(self, loader_state: PyTree) -> tuple[PyTree[Array], PyTree]:
        """
        Main logic to load a single batch of time series data segments.

        loader_state contains any extra state necessary to generate a particular batch:
        For random sampling, this corresponds to the random key, and for minibatch
        sampling, this would correspond to the batch index (and the random key if the
        data is reshuffled each epoch.)
        """
        (batch_state,) = loader_state
        linear_indices, batch_state_next = self.batch_strategy.generate_batch(
            batch_state
        )
        batch = self.get_segments(*self.linear_to_sample_indices(linear_indices))
        loader_state_next = (batch_state_next,)
        return batch, loader_state_next
