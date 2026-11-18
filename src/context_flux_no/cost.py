"""Parameter counts, FLOP counts, memory and run time of the trained models.

The functions answer the cost questions of the ICLR 2027 reviews (rebuttal plan,
action 4b): how many parameters each part of a model has, how many of them the
hypernetwork generates, and what one step costs in FLOPs, memory and time.
`scripts/evaluation/compute_cost.py` drives them on the submitted checkpoints.
"""

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .models.multiphysics import (
    ContextAppendedFluxNO,
    DISCO,
    DPOT,
    HyperFluxFNOLocal,
    HyperNeuralOperator,
)
from .models.multiphysics.abstract import AbstractMultiphysicsOperator
from .utils import num_parameters


@dataclasses.dataclass(frozen=True)
class Measurement:
    """The cost of one compiled program, as XLA reports it and as it runs.

    **Attributes:**

    - `flops`: The FLOPs that XLA counts in the program. A loop body is counted once,
        whatever the number of iterations.
    - `argument_bytes`: The size of the inputs of the program on the device.
    - `output_bytes`: The size of its outputs.
    - `temp_bytes`: The size of the buffers that the program allocates while it runs.
    - `seconds`: The wall-clock time of each timed call.
    - `peak_bytes`: The peak memory in use on the device since the process started,
        read after the timed calls. NaN on a device without memory statistics, such as
        the CPU.
    """

    flops: float
    argument_bytes: int
    output_bytes: int
    temp_bytes: int
    seconds: Float[np.ndarray, " repeats"]
    peak_bytes: float


def _compile(fn: Callable, *args: Any):
    """Compile `fn` for `args` and return the executable and the arrays it takes.

    Every non-array leaf of `args`, such as the activation function of a model, is
    closed over, and the same leaves are dropped from the output, so that `jax.jit`
    sees arrays only. A function that returns a model, such as a training step, then
    returns its arrays.
    """
    dynamic, static = eqx.partition(args, eqx.is_array)
    jitted = jax.jit(lambda d: eqx.filter(fn(*eqx.combine(d, static)), eqx.is_array))
    return jitted.lower(dynamic).compile(), dynamic


def _flops_of(compiled) -> float:
    analysis = compiled.cost_analysis()
    # Older JAX versions return one dictionary per device.
    if isinstance(analysis, list):
        analysis = analysis[0]
    return float(analysis.get("flops", np.nan))


def count_flops(fn: Callable, *args: Any) -> float:
    """Return the FLOPs that XLA counts in `fn(*args)`.

    The count is that of the compiled program on the default device. A loop body is
    counted once, so a function that loops, such as a rollout or an adaptive ODE
    solve, is counted one iteration at a time.
    """
    return _flops_of(_compile(fn, *args)[0])


def measure(fn: Callable, *args: Any, repeats: int, warmup: int) -> Measurement:
    """Compile `fn(*args)` once, then time `repeats` calls after `warmup` calls.

    Each call is waited on until its outputs are ready, so the time includes the whole
    computation on the device and not only its dispatch.

    **Arguments:**

    - `fn`: The function to measure. Its non-array arguments are closed over.
    - `args`: Its arguments.
    - `repeats`: The number of timed calls.
    - `warmup`: The number of untimed calls before them.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError(f"Need repeats >= 1 and warmup >= 0, got {repeats}, {warmup}.")
    compiled, dynamic = _compile(fn, *args)
    memory = compiled.memory_analysis()
    for _ in range(warmup):
        jax.block_until_ready(compiled(dynamic))
    seconds = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        out = jax.block_until_ready(compiled(dynamic))
        seconds[i] = time.perf_counter() - start
    (device,) = jax.tree.leaves(out)[0].devices()
    stats = device.memory_stats()
    peak = np.nan if stats is None else float(stats.get("peak_bytes_in_use", np.nan))
    return Measurement(
        flops=_flops_of(compiled),
        argument_bytes=int(memory.argument_size_in_bytes),
        output_bytes=int(memory.output_size_in_bytes),
        temp_bytes=int(memory.temp_size_in_bytes),
        seconds=seconds,
        peak_bytes=peak,
    )


def parameter_breakdown(model: AbstractMultiphysicsOperator) -> dict[str, int]:
    """Count the trainable parameters of each part of a model.

    Every model reports `total`. The in-context models with a hypernetwork also report
    `generated`, the size of the network whose parameters the hypernetwork outputs.
    That network is not part of `total`. `other` is what `total` holds beyond the
    named parts.

    **Arguments:**

    - `model`: A DPOT, DISCO, HFluxNO (`HyperFluxFNOLocal` or `HyperNeuralOperator`) or
        context-concatenation (`ContextAppendedFluxNO`) model.
    """
    total = num_parameters(model)
    if isinstance(model, HyperFluxFNOLocal | HyperNeuralOperator):
        heads = (
            model.hypernetwork_heads
            if isinstance(model, HyperFluxFNOLocal)
            else (model.hypernetwork_head,)
        )
        embedding = jnp.zeros(model.embedding_dim)
        parts = {
            "encoder": num_parameters(model.context_encoder),
            "trunk": num_parameters(model.hypernetwork_trunk),
            "head": sum(num_parameters(head.linears) for head in heads),
        }
        extra = {"generated": sum(num_parameters(head(embedding)) for head in heads)}
    elif isinstance(model, ContextAppendedFluxNO):
        parts = {
            "encoder": num_parameters(model.context_encoder),
            "trunk": num_parameters(model.hypernetwork_trunk),
            "flux": num_parameters(model.fluxes),
        }
        extra = {}
    elif isinstance(model, DISCO):
        parts = {
            "encoder": num_parameters(model.hypernet),
            "head": num_parameters((model.mlp_params_common, model.mlp_params_heads)),
        }
        extra = {"generated": num_parameters(model.operatornet)}
    elif isinstance(model, DPOT):
        parts, extra = {}, {}
    else:
        raise ValueError(f"No parameter breakdown for {type(model).__name__}.")
    return {"total": total, **parts, "other": total - sum(parts.values()), **extra}


def step_flop_breakdown(
    model: AbstractMultiphysicsOperator,
    context: Float[Array, " time channels *spatial_dims"],
    args: tuple[float, ...],
) -> dict[str, float]:
    """Count the FLOPs of one step of a model for one context, and of its parts.

    Every model reports `step`, one call of the model. The other entries depend on the
    model:

    - `HyperFluxFNOLocal`: `generation`, the encoder and the hypernetwork, and
        `flux_application`, one finite-volume step with generated flux networks. Their
        sum is `step`. `hypernetwork_head` is the part of `generation` that outputs
        the parameters.
    - `HyperNeuralOperator`: `encoder` and `hypernetwork_head`. The rest of `step` is
        the trunk and the target network.
    - `DISCO`: `operator_call`, one evaluation of the operator network. Its adaptive
        ODE solve evaluates the network several times per step, while `step` counts
        one solver iteration.

    **Arguments:**

    - `model`: The model, in inference mode.
    - `context`: One context window.
    - `args`: The `args` of the model, such as `(dt, dx)`.
    """
    flops = {"step": count_flops(lambda m, u: m(u, args)[0], model, context)}
    if isinstance(model, HyperFluxFNOLocal | HyperNeuralOperator):
        heads = (
            model.hypernetwork_heads
            if isinstance(model, HyperFluxFNOLocal)
            else (model.hypernetwork_head,)
        )
        flops["hypernetwork_head"] = count_flops(
            lambda hs, c: [h(c) for h in hs],
            heads,
            jnp.zeros(model.embedding_dim),
        )
    if isinstance(model, HyperFluxFNOLocal):
        flops["generation"] = count_flops(
            lambda m, u: m.generate_solver(u),
            model,
            context,
        )
        flux_models = model.generate_solver(context)
        flops["flux_application"] = count_flops(
            lambda m, f, u: m.step(f, u, args), model, flux_models, context[-1]
        )
    elif isinstance(model, HyperNeuralOperator):
        # `HyperNeuralOperator.__call__` appends grid channels before the encoder when
        # `stack_grid` is set. The 2D Euler model of the submission does not set it,
        # so the encoder reads the context as it is.
        if model.stack_grid:
            raise ValueError("The encoder FLOPs assume stack_grid=False.")
        flops["encoder"] = count_flops(
            lambda m, u: m(u), model.context_encoder, context
        )
    elif isinstance(model, DISCO):
        flops["operator_call"] = count_flops(
            lambda net, u: net(u), model.operatornet, context[-1]
        )
    elif not isinstance(model, DPOT | ContextAppendedFluxNO):
        raise ValueError(f"No FLOP breakdown for {type(model).__name__}.")
    return flops
