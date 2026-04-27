from typing import Mapping

import jax
from jax import jit, random

from ..core.train import TrainState


@jit
def train_step(
    rng: jax.Array,
    state: TrainState,
    batch: Mapping,
    **kwargs,
):
    rng_dropout, rng_extra = random.split(rng)
    rngs = {"dropout": rng_dropout, "extra": rng_extra}
    x, theta = batch["x"], batch["theta"]

    def loss_fn(params):
        output = state.apply_fn({"params": params}, x, training=True, rngs=rngs)
        return output.nll(theta)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: Mapping,
    **kwargs,
):
    x, theta = batch["x"], batch["theta"]
    rngs = {"extra": rng}
    output = state.apply_fn({"params": state.params}, x, training=False, rngs=rngs)
    return output.metrics(theta)
