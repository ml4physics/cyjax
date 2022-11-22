from __future__ import annotations

import jax
import jax.numpy as jnp

from chex import Shape, Scalar, Array, PRNGKey
from typing import Union
from .projective import index_hom_to_affine, hom_to_affine


def uniform_components(key: PRNGKey, shape: Shape) -> Array:
    """Complex values with uniform real and imaginary parts.

    Imaginary and real components are individually uniformly distributed
    between 0 and 1. This results in uniform values in the unit square
    in the complex plane.

    Example:
        >>> x = uniform_components(jax.random.PRNGKey(0), (4,))
        >>> jnp.all(jnp.abs(x.real) < 1)
        DeviceArray(True, dtype=bool)
        >>> jnp.all(jnp.abs(x.imag) < 1)
        DeviceArray(True, dtype=bool)

    Args:
        key: PRNG random key.
        shape: tuple of nonnegative integers.
    Returns:
        Array of requested ``shape``.
    """
    r, i = jax.random.uniform(key, (2, *shape), float, -1, 1)
    return r + 1j * i


def uniform_angle(key: PRNGKey, shape: Shape,
                  minrad: Scalar = 0, maxrad: Scalar = 1) -> Array:
    """Complex values individually uniform on complex disk.

    Radius of complex valuse lie between ``minrad``
    (default 0, corresponding to a disk) and ``maxrad``.

    Example:
        >>> x = uniform_angle(jax.random.PRNGKey(0), (4,))
        >>> jnp.all(jnp.abs(x) < 1)
        DeviceArray(True, dtype=bool)

    Args:
        key: PRNG random key.
        shape: Tuple of nonnegative integers.
        minrad: Minimal absolute value of generated values.
        maxrad: Maximal absolute value of generated values.
    Returns:
        Array of requested ``shape`` containing complex values.
    """
    rand_a, rand_r = jax.random.uniform(key, (2, *shape))
    rad = jnp.sqrt(minrad**2 + rand_r * (maxrad**2 - minrad**2))
    return rad * jnp.exp(2j * jnp.pi * rand_a)


def uniform_angle_unit(key: PRNGKey, shape: Shape):
    """Complex values with unit modulus and uniform angle.

    Example:
        >>> x = uniform_angle_unit(jax.random.PRNGKey(0), (4,))
        >>> jnp.allclose(jnp.abs(x), 1)
        DeviceArray(True, dtype=bool)

    Args:
        key: PRNG random key.
        shape: Tuple of nonnegative integers.
    Returns:
        Array of requested ``shape`` containing complex values.
    """
    return jnp.exp(2j * jnp.pi * jax.random.uniform(key, shape))


def real_sphere(key: PRNGKey, shape: Shape):
    r"""Random points on real unit sphere.

    The last value in the shape tuple is interpreted as dimension :math:`n`
    such that the points lie on the sphere :math:`S^{n-1}` of unit radius
    in :math:`\mathbb{R}^n`.

    Example:
        >>> x = real_sphere(jax.random.PRNGKey(0), (4, 3))
        >>> jnp.allclose(jnp.linalg.norm(x, axis=-1), 1)
        DeviceArray(True, dtype=bool)

    Args:
        key: PRNG random key.
        shape: Tuple of nonnegative integers. Last value is interpreted
            as dimension of (ambient) real vector space.
    Returns:
        Array of requested ``shape`` containing real values.
    """
    if len(shape) == 0:
        return jnp.array(1, dtype=float)
    points = jax.random.normal(key, shape)
    return points / jnp.linalg.norm(points, axis=-1, keepdims=True)


def complex_sphere(key: PRNGKey, shape: Shape):
    r"""Sample uniform points on the unit sphere in complex space..

    The last value in the shape tuple is interpreted as dimension :math:`n`
    such that the points lie on the sphere :math:`S^{2n-1}` of unit radius
    in :math:`\mathbb{C}^n`.

    Example:
        >>> x = complex_sphere(jax.random.PRNGKey(0), (4, 3))
        >>> jnp.allclose(jnp.linalg.norm(x, axis=-1), 1)
        DeviceArray(True, dtype=bool)

    Args:
        key: PRNG random key.
        shape: Tuple of nonnegative integers. Last value is interpreted
            as dimension of (ambient) complex vector space.
    Returns:
        Complex array of requested ``shape``.
    """
    if len(shape) == 0:
        return jnp.array(1, dtype=complex)
    dim = shape[-1]
    pts = real_sphere(key, (*shape[:-1], 2 * dim))
    return pts[..., :dim] + 1j * pts[..., dim:]


def uniform_projective(
        key: PRNGKey, count: int, dim: int, affine: bool = True) \
        -> Union[Array, tuple[Array, Array]]:
    r"""Uniformly distributed points in complex projective space.

    Generate uniform points in :math:`\mathbb{P}^{\mathrm{dim}}`.
    If ``affine`` is true, return a tuple of affine coordinates and
    the respective patch index.
    Otherwise return the points as homogeneous coordinates which consist
    of :math:`\mathrm{dim}+1` complex numbers.

    Args:
        key: PRNG random key.
        count: Number of points to generate.
        dim: Dimension of projective space.
        affine: Whether to return the points as affine coordinates plus
            patch index, as opposed to homogeneous coordinates.
    Returns:
        Array of homogeneous coordinates with shape (count, dim+1) or
        tuple of affine coordinates (count, dim) and patches (count,).
    """
    zs_hom = complex_sphere(key, (count, dim + 1))
    if affine:
        return hom_to_affine(zs_hom)
    else:
        return zs_hom


def projective_overlap(
        key: PRNGKey, count: int, dim: int,
        this_patch: int, other_patch: int, patch_size: Scalar):
    r"""Sample points lying in overlap region of two affine patches.

    The overlap region is defined such that the coordinate values
    do not exceed ``patch_size`` in either patch,
    i.e. :math:`z^i \leq \mathrm{patch\_size}`.
    The points are sampled such that the distribution is invariant
    under :func:`cyjax.change_chart`.

    Args:
        key: PRNG random key.
        count: Number of random points to generate.
        dim: Dimension of complex projective space.
        this_patch: Patch to sample affine coordinates in.
        other_patch: Generate points in overlap region with this patch.
    Returns:
        Shape :math:`\mathrm{count} \times \mathrm{dim}`
        array of affine coordinates which lie in the overlap region.
    """
    if count == 0 or dim == 0:
        return jnp.empty((count, dim), complex)

    # local index of other patch
    other = index_hom_to_affine(this_patch, other_patch)

    rand_r, rand_a = jax.random.uniform(key, (2, count, dim))
    # symmetric sampling, 1 / patch_size < rad_0 < patch_size
    log_rad_min = jnp.log(1 / patch_size)
    rad_0 = jnp.exp(
        log_rad_min + (jnp.log(patch_size) - log_rad_min) * rand_r[:, 0])

    # remaining radii < min(patch_size, rad_max * patch_size)
    rad_max = jnp.minimum(patch_size * rad_0, patch_size)
    # set all and then correct the other index
    rad_all = rad_max.reshape(-1, 1) * jnp.sqrt(rand_r)
    rad_all = rad_all.at[:, other].set(rad_0)

    return rad_all * jnp.exp(2j * jnp.pi * rand_a)
