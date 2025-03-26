from typing import Callable

import flax.linen as nn

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput


class NPE(nn.Module):
    estimator: Callable = MLP([64, 64, 2])
    output_fn: Callable = DiagonalMVNOutput.from_activations

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.output_fn(self.estimator(x))
