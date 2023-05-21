# Copyright 2022 Mathis Gerdes
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of machine learning functions and classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
import numpy as np
import flax.linen as nn

from functools import partial
from .util import PRNGSequence
from typing import Sequence, Union, Callable, Optional
from itertools import product as _cart


@jax.jit
def cholesky_decode(diag: chex.Array, upper: chex.Array):
    r"""Construct hermitian matrix from Cholesky decomposition.

    From ``diag`` and ``upper``, first a matrix :math:`M` is initialized as

    .. math::
        M = \begin{bmatrix}
            \mathrm{diag}_1 & & \text{upper}_{ij}\\
            & \ddots & \\
            0 &  & \mathrm{diag}_n
        \end{bmatrix}

    The output of this function is then :math:`M M^\dagger`,
    which is Hermitian and positive-definite.
    Indexing of the 1d `diag` array with respect to the upper triangular
    is done as in :func:`np.triu_indices` (i.e. row-major).

    Args:
        diag: Real 1d-array of diagonal entries.
        upper: Complex 1d-array of upper diagonal entries.
    Returns:
        Hermitian, positive-definite matrix constructed by the
        Cholesky decomposition.
    """
    mat = jnp.diag(diag.astype(upper.dtype))
    upper_ind = jnp.triu_indices(diag.shape[0], 1)
    mat = mat.at[upper_ind].set(upper)
    return mat @ mat.T.conj()


@jax.jit
def cholesky_from_param(h_par):
    """Construct hermitian matrix from Cholesky decomposition parameters.

    This is a wrapper around :func:`cholesky_decode` which takes
    a single array of real values as input. These comprise the real diagonal
    entries and the real and imaginary parts of the upper triangular entries.

    Args:
        h_par: Real 1d-array. This is simply a concatenation of `diag`
            and `upper` arguments of :func:`cholesky_decode`.
    Returns:
        Hermitian, positive-definite matrix constructed by the
        Cholesky decomposition.
    """
    basis_size = np.round(np.sqrt(h_par.size)).astype(int)
    diag = h_par[:basis_size]
    upper_r, upper_i = h_par[basis_size:].reshape(2, -1)

    return cholesky_decode(diag, upper_r + 1j * upper_i)


def hermitian_param_init(key: chex.PRNGKey, basis_size: int,
                         fluctuate: chex.Scalar = None, dtype=float):
    """Initialize parametrization to yield identity for Hermitian matrix.

    The returned parameters will give a positive-definite Hermitian matrix
    when transformed by :func:`cholesky_from_param`

    If ``fluctuate`` is given, diagonal is set to uniform random values
    between ``1 - fluctuate`` and ``1 + fluctuate``.
    """
    par = jnp.zeros((basis_size ** 2,), dtype=dtype)

    if fluctuate is not None:
        diag = 1 + jax.random.normal(key, (basis_size,)) * fluctuate
    else:
        diag = 1

    par = par.at[:basis_size].set(diag)
    return par


class BatchSampler:
    def __init__(self,
                 seed,
                 variety,
                 params_sampler: Callable[[chex.PRNGKey, int], chex.Array],
                 batch_size_params: int = 5,
                 batch_size: int = 100,
                 buffer_size: int = 20,
                 backend='cpu',
                 device=None):
        """Iterable buffered sample generator.

        Once initialized, new samples can be obtained as
        >>> params, zs, patch, weights = next(batch_sampler)

        Note that creating the sampler can take several seconds because
        the sampling function is jit-compiled and the first batch is sampled.

        Args:
            seed: Random key or integer.
            variety: Variety to sample for.
            params_sampler: Function which samples new complex moduli
                values given a random key and an integer batch size.
            batch_size_params: Number of complex moduli values to sample
                for each batch.
            batch_size: Number of points on the variety per moduli value.
            buffer_size: Number of samples to keep in buffer.
            backend: Backend used for sampling.
            device: Device used for training. Samples are transferred to this
                device after they are generated.
        """
        self.buffer_size = buffer_size
        if device is None:
            device = jax.devices()[0]
        self.device = device

        @jax.vmap
        def _sample(key, params):
            """Sample new points."""
            # for each set of parameters, sample `batch_size` points
            return variety.sample_intersect(key, params, batch_size, affine=True, weights=True)

        @partial(jax.jit, backend=backend)
        def sample(key):
            """Sample new parameters and points."""
            k1, k2 = jax.random.split(key)
            params_batch = params_sampler(k1, batch_size_params)
            keys = jax.random.split(k2, batch_size_params)
            (zs, patch), weights = _sample(keys, params_batch)
            return params_batch, zs, patch, weights

        self.sample = sample
        self.rns = PRNGSequence(seed)
        self.buffer = []

        self.new_buffer()
        self._index = 0

    def new_buffer(self):
        """Generate and buffer new samples."""
        self.buffer.clear()
        for b in range(self.buffer_size):
            sample = self.sample(next(self.rns))
            # moving arrays to device here is crucial for performance
            sample = tuple(jax.device_put(a, self.device) for a in sample)
            self.buffer.append(sample)

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.buffer[self._index]
        self._index += 1
        if self._index == self.buffer_size:
            self.new_buffer()
            self._index = 0
        return sample


def variance_eta_loss(h, sample, algebraic_metric):
    """Compute variance-based eta loss."""
    # here, psi is a single set of moduli
    params, zs, patch, weights = sample

    etas = algebraic_metric.eta(h, zs, params, patch).real
    eta_mean = jnp.mean(weights * jax.lax.stop_gradient(etas)) / jnp.mean(weights)

    loss = (etas / eta_mean - 1) ** 2
    loss *= weights
    loss = jnp.mean(loss)

    # if g is not pos. def. eta may be negative -> penalty
    loss += jnp.mean(jnp.log(jnp.where(etas < 0, etas, 0) ** 2 + 1))

    return loss


class HNetMLP(nn.Module):
    r"""Dense network for learning moduli dependence of the H matrix.

    Given moduli :math:`\psi_i` as inputs, first a number of features
    are built form this which are then fed into a chosen number of dense linear
    layers. Finally, a linear layer without activation function is used to
    construct the parameters of the H matrix. If ``sig_suppress`` is true,
    two copies are output in this way and the final outputs are
    ``out[0] * sigmoid(out[1])``.

    The input features are constructed as follows.
    First, raise each moduli parameter to all chosen powers :math:`p_j`,
    :math:`\psi_i^{p_j}`. Then, take all possible products of these
    (in the case of multiple moduli). Calling the resulting products
    :math:`f_n`, the input features are a concatenation of

    - :math:`|f_n|` if ``feature_abs`` is true.
    - :math:`arg(f_n)` if ``feature_angle`` is true.
    - :math:`Re[f_n]` and :math:`Im[f_n]` if ``feature_parts`` is true.

    Note that 0 is always included as a power if there are multiple moduli,
    even if not explicitly chosen.

    Args:
        basis_size: Number of sections used.
        layer_sizes: Number of hidden dense layers.
        dropout_rates: Dropout rates per layer. Either None, or a list
            of floats/None where the latter indicates no dropout.
        powers: Powers to use to extract features from moduli parameter inputs.
        sig_suppress: Whether to learn an output which is transformed by
            sigmoid and multiplies the entries of the H-matrix.
        init_fluctuation: Initial fluctuation of the diagonal around 1.
        activation: Activation function for hidden layers.
        feature_angle: Whether to include the complex angle of the moduli
            powers as feature.
        feature_abs: Whether to include the absolute value of the moduli powers
            as features.
        feature_parts: Whether to use real and imaginary parts of moduli powers
            as features.
    """
    #: Number of sections used.
    basis_size: int
    #: Number of hidden dense linear layers.
    layer_sizes: Sequence[int]
    #: Dropout rates per layer.
    dropout_rates: Sequence[Optional[float]] = None
    #: Powers to use to extract features from moduli parameter inputs.
    powers: Sequence[Union[int, float]] = (1, 2, 3)
    #: Whether to use learnable sigmoid-suppression of matrix elements.
    sig_suppress: bool = True
    #: Fluctuation of initialization around the identity (of diagonal entries).
    init_fluctuation: float = 1e-3
    #: Activation function for hidden layers.
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.sigmoid
    # features
    #: Whether to use complex angle as input feature.
    feature_angle: bool = True
    #: Whether to use absolute values as input feature.
    feature_abs: bool = True
    #: Whether to use real & imaginary part as input feature.
    feature_parts: bool = True

    @nn.compact
    def __call__(self, moduli_params, deterministic=False):
        moduli_params = jnp.atleast_1d(moduli_params)
        if moduli_params.ndim == 1:
            single = True
            moduli_params = moduli_params.reshape(1, -1)
        elif moduli_params.ndim == 2:
            single = False
        else:
            raise ValueError('Input must be scalar, one, or two dimensional.')

        powers = np.unique([0, *self.powers])
        
        psi_pow = []
        for pw in _cart(*([powers] * moduli_params.shape[-1])):
            pw = np.array(pw)
            if not np.all(pw == 0):
                product = jnp.prod(jnp.power(moduli_params, pw), -1)
                psi_pow.append(product)
        psi_pow = jnp.stack(psi_pow, -1)

        x = jnp.concatenate(
            ([jnp.angle(moduli_params)] if self.feature_angle else []) +
            ([jnp.abs(psi_pow)] if self.feature_abs else []) +
            ([psi_pow.real, psi_pow.imag] if self.feature_parts else []),
            axis=1)

        dropout = self.dropout_rates
        if dropout is None:
            dropout = [None] * len(self.layer_sizes)
        for features, rate in zip(self.layer_sizes, dropout):
            x = nn.Dense(features, dtype=x.dtype)(x)
            x = self.activation(x)
            if rate is not None:
                x = nn.Dropout(rate=rate)(x, deterministic=deterministic)

        if self.sig_suppress:
            def bias_init(k, s, d):
                h = hermitian_param_init(
                    k, self.basis_size, self.init_fluctuation)
                return jnp.concatenate((jnp.zeros_like(h), h), axis=0)
        else:
            def bias_init(k, s, d):
                return hermitian_param_init(
                    k, self.basis_size, self.init_fluctuation)

        # final linear layer to H-parameters
        h_features = self.basis_size ** 2
        if self.sig_suppress:
            h_features *= 2
        h_params = nn.Dense(
            h_features,
            name='final_dense',
            dtype=x.dtype,
            # initialize such that H starts close to the identity
            bias_init=bias_init,
            kernel_init=nn.initializers.constant(0., dtype=x.dtype),
        )(x)

        if self.sig_suppress:
            out = h_params.reshape(-1, 2, self.basis_size ** 2)
            h_params = jax.nn.sigmoid(out[:, 0]) * out[:, 1]

        if single:
            return jnp.squeeze(h_params, 0)
        return h_params
