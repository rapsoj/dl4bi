from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jax.tree_util import Partial


class ResidualEmbedding(nn.Module):
    """Returns [x, embed(x)]."""

    embed: Callable

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False):
        return jnp.concatenate([x, self.embed(x, training)], axis=-1)


class FixedSinusoidalEmbedding(nn.Module):
    r"""Fixed sinusoidal positional encoding from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    $$
    \begin{aligned}
        \text{pe}(s,2i)&=\frac{s}{\text{max-len}^{2i/d}} \\\\
        \text{pe}(s,2i+1)&=\frac{s}{\text{max-len}^{2i/d}}
    \end{aligned}
    $$

    .. warning::
        This maps each element of the last dimension independently and then
        concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times
        DE}$. For example, a 3 dimensional point `(x,y,z)` will get mapped to a
        `3*embed_dim` embedding.
    """

    embed_dim: int = 256
    max_len: int = 10000

    @nn.compact
    def __call__(self, s: jax.Array, training: bool = False):
        B, L, D = s.shape
        return _pe_attn_sinusoidal(
            self.embed_dim,
            self.max_len,
        )(s).reshape(B, L, D * self.embed_dim)


def _pe_attn_sinusoidal(d: int, max_len: float = 10000):
    f = lambda i, s: s / (max_len ** (2 * i / d))
    return _pe_sinusoidal(f, d)


# TODO(danj): period makes a huge difference here...
def _pe_nerf_sinusoidal(d: int):
    f = lambda i, s: 2**i * jnp.pi * s
    return _pe_sinusoidal(f, d)


def _pe_sinusoidal(f: Callable, d: int):
    i = jnp.arange(d // 2)
    vf = lambda s: jnp.apply_along_axis(vmap(Partial(vmap(f, (0, None)), i)), -1, s)
    return jit(lambda s: jnp.concatenate([jnp.sin(vf(s)), jnp.cos(vf(s))], -1))


class NeRFEmbedding(nn.Module):
    r"""Positional encoding with MLP from ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

    $$
    \begin{aligned}
        F_\Theta&=F_\Theta^\prime\circ\gamma \\\\
        \gamma(s)&=[\sin(2^0\pi s), \cos(2^0\pi s),\ldots,\sin(2^{d-1}\pi s),\cos(2^{d-1}\pi s)] \\\\
    \end{aligned}
    $$

    .. warning::
        This maps each element of the last dimension independently and then
        concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times
        DE}$. For example, a 3 dimensional point `(x,y,z)` will get mapped to a
        `3*embed_dim` embedding.
    """

    embed_dim: int = 256

    @nn.compact
    def __call__(self, s: jax.Array, training: bool = False):
        B, L, D = s.shape
        return _pe_nerf_sinusoidal(self.embed_dim)(s).reshape(B, L, D * self.embed_dim)


class GaussianFourierEmbedding(nn.Module):
    r"""Gaussian Fourier Feature (GFF) positional encoding from ["Fourier Features Let Networks Learn
        High Frequency Functions in Low Dimensional Domains"](https://arxiv.org/abs/2006.10739).

    $$
    \begin{aligned}
        \gamma(\mathbf{v})&=[\cos (2 \pi \mathbf{B v}), \sin (2 \pi \mathbf{B} \mathbf{v})] \\\\
        \mathbf{B}&\sim\mathcal{N}(0, \sigma^2)
    \end{aligned}
    $$

    .. warning::
        This maps every element of the last dimension together and produces
        both a sine and cosine feature: $\mathbb{R}^{\ldots\times D}
        \to\mathbb{R}^{\ldots\times E}$.
    """

    embed_dim: int = 256
    std: float = 4.0

    @nn.compact
    def __call__(self, s, training: bool = False):
        gen_B = lambda rng: random.normal(rng, (self.embed_dim // 2, s.shape[-1]))
        B = self.variable("projections", "B", lambda: gen_B(self.make_rng("params")))
        return _pe_gaussian_fourier(B.value, self.std)(s)


def _pe_gaussian_fourier(B: jax.Array, std: float):
    s_proj = lambda s: (2.0 * jnp.pi * s) @ (std * B).T
    return jit(
        lambda s: jnp.concatenate([jnp.sin(s_proj(s)), jnp.cos(s_proj(s))], axis=-1)
    )


class RFF_FeaturesEncoder(nn.Module):
    D: int

    @nn.compact
    def __call__(self, s: jax.Array):
        n_features = s.shape[-1]
        w = self.variable(
            "projections",
            "rff_w",
            lambda: jax.random.normal(
                self.make_rng("params"), shape=(n_features, self.D)
            )
            * jnp.sqrt(2),
        )
        b = self.variable(
            "projections",
            "rff_b",
            lambda: jax.random.uniform(
                self.make_rng("params"), shape=(self.D,), minval=0, maxval=2 * jnp.pi
            )
            * jnp.sqrt(2),
        )
        projection = jnp.dot(s, w.value) + b.value  # (n_samples, D)
        rff = jnp.sqrt(2.0 / self.D) * jnp.cos(projection)
        return rff
