import abc

import equinox as eqx
import jax
from jaxtyping import Array, Int, PyTree


BatchState = PyTree


class AbstractBatching(eqx.Module):
    batch_size: eqx.AbstractVar[int | None]

    @abc.abstractmethod
    def init(self, num_total_data: int) -> BatchState: ...

    @abc.abstractmethod
    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        pass


class RandomMiniBatching(AbstractBatching):
    batch_size: int = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)

    def __init__(self, batch_size: int, *, random_seed: int = 0):
        self.batch_size = batch_size
        self.random_seed = random_seed

    def init(self, num_total_data: int) -> BatchState:
        return num_total_data, jax.random.key(self.random_seed)

    def generate_batch(
        self, batch_state: BatchState
    ) -> tuple[Int[Array, " {self.batch_size}"], BatchState]:
        num_total_data, key = batch_state
        key, key_next = jax.random.split(key)
        batch_data_indices = jax.random.randint(
            key, (self.batch_size,), 0, num_total_data
        )
        batch_state_next = (num_total_data, key_next)
        return batch_data_indices, batch_state_next
