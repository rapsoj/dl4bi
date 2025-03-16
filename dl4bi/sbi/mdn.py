import flax.linen as nn
import jax

from ..core.mlp import MLP
from ..core.model_output import MDNOutput


class MLPMDN(nn.Module):
    k: int = 5
    num_hidden: int = 128
    num_layers: int = 3

    @nn.compact
    def __call__(self, x: jax.Array, **kwargs):
        params = MLP([self.num_hidden] * self.num_layers + [3 * self.k])(x)
        return MDNOutput.from_activations(params)
