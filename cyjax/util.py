from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from math import factorial
from typing import Union, Callable, Generator
from chex import Numeric, Array, Shape, PRNGKey


# make backward compatible for python < 3.8 (insted of math.prod)
def prod(factors: Union[np.ndarray, Array, Generator]) \
        -> Union[np.ndarray, Array]:
    pr = None
    for factor in factors:
        if pr is None:
            pr = factor
        else:
            pr *= factor
    return np.ones(()) if pr is None else pr


@jax.jit
def pop(arr: Array, index: int) -> tuple[Array, Array]:
    """Pop value of arr at index in JIT-compatible way.

    Args:
        arr: An array.
        index: Remove element at this index, ``arr[index]``, from the array.
    Returns:
        A tuple of the removed value and the remaining array.
    """
    chex.assert_type(index, int)
    rolled = jnp.roll(arr, -index, axis=0)
    return rolled[0], jnp.roll(rolled[1:], index, axis=0)


@jax.jit
def insert_1d(arr: Array, val: Numeric, index: int) -> Array:
    """Insert value into array at given index.

    If index is negative, counted backwards and with respect to
    the generated output array.

    Args:
        arr: 1-dimensional array.
        index: Single scalar index.
        val: 0- or 1- dimensional array.

    Returns:
        Combined array.
    """
    val = jnp.atleast_1d(val)
    chex.assert_rank(arr, 1)
    chex.assert_rank(val, 1)
    shift = -jnp.where(index < 0, index + 1, index)
    shift_back = jnp.where(index < 0, -shift, index + len(val))
    shifted = jnp.roll(arr, shift)
    val = val.flatten()
    return jnp.roll(jnp.append(shifted, val, 0), shift_back)


@jax.jit
def insert_col(mat: Array, col: Array, index: int) -> Array:
    """Insert a column or column of sub-rows into a 2D matrix.

    Args:
        mat: Original 2-dimensional array.
        col: New column to insert.
        index: New index of column to be inserted.

    Returns:
        Combined array.
    """
    chex.assert_rank(mat, 2)
    if col.ndim == 1:
        col = col.reshape(-1, 1)
    chex.assert_rank(col, 2)

    shift = -jnp.where(index < 0, index + 1, index)
    shift_back = jnp.where(index < 0, -shift, index + col.shape[1])
    mat = jnp.roll(mat, shift, axis=1)
    mat = jnp.concatenate([mat, col], axis=1)
    return jnp.roll(mat, shift_back, axis=1)


def shuffle_axis(key: PRNGKey, x: Array, axis: int) -> Array:
    """Shuffle indices of one axis.

    Args:
        key: PRNG random key.
        x: Array of values.
        axis: Axis to shuffle.

    Returns:
        New array with values shuffled along given axis.
    """
    n = int(np.prod(x.shape[:axis]))
    x_ = x.reshape(n, *x.shape[axis:])
    perm = jax.vmap(jax.random.permutation)(jax.random.split(key, n), x_)
    return perm.reshape(x.shape)


def binomial(n: int, k: int) -> int:
    r"""Binomial coefficients (cannot apply JIT).

    Returns:
        Given ``n`` and ``k`` returns :math:`\frac{n!}{k! (n-k)!}`.
    """
    return factorial(n) // (factorial(k) * factorial(n-k))


def flatten_coord(zs: Array, zs_c: Array = None, need_c: bool = False) \
        -> tuple[Union[Array, tuple[Array, Array]], tuple[int, ...]]:
    """Flatten coordinates and return old shape.

    Args:
        zs: Array of complex coordinates.
        zs_c: Complex conjugate of ``zs``.
        need_c: Whether to return (and compute) the complex conjugate.
    Returns:
        Tuple of flattened coordinates and old_shape.
        If ``need_c`` is true, coordinates are a tuple of two arrays,
        otherwise it is one array.
    """
    zs = jnp.atleast_1d(zs)
    old_shape = zs.shape[:-1]  # omit dimension
    dim = zs.shape[-1]
    if need_c:
        if zs_c is None:
            zs_c = zs.conj()
        else:
            zs_c = jnp.atleast_1d(zs_c)
        chex.assert_equal_shape((zs, zs_c))
        return (zs.reshape((-1, dim)), zs_c.reshape((-1, dim))), old_shape
    else:
        return zs.reshape((-1, dim)), old_shape


def unflatten_coord(arr: Union[Array, tuple[Array, ...]], old_shape: Shape) \
        -> Union[Array, tuple[Array, ...]]:
    """Bring output arrays to old shape before flatten_coord."""
    if isinstance(arr, jnp.ndarray):
        return arr.reshape((*old_shape, *arr.shape[1:]))
    else:
        return jax.tree_util.tree_map(
            lambda a: a.reshape((*old_shape, *a.shape[1:])), arr)


def mc_integrate(
        key: PRNGKey, count: int, fn: Callable[[Array], Array],
        sample: Callable[[PRNGKey, int], Array], var: bool = False) \
        -> Union[Numeric, tuple[Numeric, Numeric]]:
    r"""Monte Carlo integration.

    This function approximates

    ..math::
        \int \mathrm{fn}(z) \approx \sum_{z, w \sim \mathrm{sample}} \mathrm{fn}(z) \cdot w \,.

    The function ``sample`` is expected to return points :math:`zs` and
    corresponding weights :math:`w`. The latter are the ratios of the
    target integration measure and the probability measure corresponding
    to the used sampling process.

    Args:
        key: PRNG random key.
        count: Number of samples used in the MC approximation.
        fn: Integrand, taking an array of points as input.
        sample: Sampling function, taking a random key and an integer
            indicating the number of desired samples as input. Must return
            a tuple of samples and corresponding weights.
        var: Whether to return an MC approximation of the variance of
            the integration result. This assumes generated samples are i.i.d.
    Returns:
        Either the integration result, or the integration result and the
        estimated variance.
    """
    zs, w = sample(key, count)

    vals = fn(zs) * w
    val = jnp.mean(w, axis=0)
    if var:
        var = (jnp.mean(vals ** 2, axis=0) - val ** 2) / count
        return val, var
    return val


def mc_integrate_batched(key, batches, batch_size, fn, sample, var):
    """Monte carlo integration in batches.

    This function computes the MC approximation of an integral
    by summing over batches of samples.
    This is done by repeatedly calling :func:`mc_integrate`.

    Args:
        key: PRNG random key.
        batches: Number of batches.
        batch_size: Number of samples in each batch.
        fn: Integrand, taking an array of points as input.
        sample: Sampling function, taking a random key and an integer
            indicating the number of desired samples as input. Must return
            a tuple of samples and corresponding weights.
    Returns:
        MC estimate of integral.
    """
    integ = partial(
        mc_integrate, sample=sample, var=var, fn=fn)

    def step(key, _):
        k1, k2 = jax.random.split(key)

        zs, w = sample(key, batch_size)

        vals = fn(zs) * w
        val = jnp.mean(w, axis=0)
        val2 = None
        if var:
            val2 = jnp.mean(vals ** 2, axis=0) / batch_size

        return k2, (val, val2)

    res, val2 = jax.lax.scan(step, key, None, batches)[1]
    mean = jnp.mean(res, axis=0)
    if var:
        variance = jnp.mean(val2) / batches - mean**2 / (batches * batch_size)
        return mean, variance
    return mean


class PRNGSequence:
    _key = None

    def __init__(self, seed: Union[chex.PRNGKey, int] = 42):
        """Random key sequence.

        Use as follows:
        >>> rns = PRNGSequence(42)
        >>> key = next(rns)
        """
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self):
        """Get the next random key."""
        k, self._key = jax.random.split(self._key)
        return k
