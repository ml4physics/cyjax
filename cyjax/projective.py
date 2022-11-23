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

from __future__ import annotations

import jax
import jax.numpy as jnp

import chex

from chex import Array
from functools import partial
from jax_autovmap import auto_vmap
from . import util
from .util import pop, insert_1d


@jax.jit
def index_hom_to_affine(patch: int, hom_index: int) -> int:
    """Convert homogeneous index to (local) affine index in given patch.

    Args:
        patch: Affine patch, i.e. index of homogeneous coordinate scaled to 1.
        hom_index: Index in array of homogeneous coordinates.
    Returns:
        Index in array of homogeneous coordinates corresponding ``hom_index``.
    """
    return jnp.where(patch <= hom_index, hom_index - 1, hom_index)


@jax.jit
def index_affine_to_hom(patch: int, index: int) -> int:
    """Get (global) homogeneous index given affine index in patch.

    Args:
        patch: Affine patch, i.e. index of homogeneous coordinate scaled to 1.
        index: Index in affine coordinate vector.
    Returns:
        Corresponding index in homogeneous coordinate vector.
    """
    return jnp.where(patch <= index, index + 1, index)


@jax.jit
@auto_vmap(1, 0, 0)
def change_chart(affine: Array, old_patch: int, new_patch: int) -> Array:
    """Convert local affine coordinates between patches.

    Args:
        affine: Array of affine coordinates.
        old_patch: Old affine patch.
        new_patch: New affine patch.
    Returns:
         Array like ``affine`` corresponding to the points but expressed
         in terms of affine coordinates of the new patch.
    """
    z0, z = pop(affine, new_patch - (old_patch < new_patch))
    return insert_1d(z / z0, 1 / z0, old_patch - (new_patch < old_patch))


@jax.jit
@auto_vmap(1, 0)
def hom_to_affine(hom: Array, patch: int = None):
    """Convert homogeneous to affine patch coordinates.

    If patch is None, go to "numerically optimal" patch,
    in which all coordinates are :math:`|z_i| < 1`.

    Args:
        hom: Homogeneous coordinates.
        patch: Affine coordinate patch to convert to.
    Returns:
        Either a tuple containing coordinates and patch index, or
        just the coordinates if a patch index is provided.
    """
    if patch is None:
        patch = jnp.argmax(jnp.abs(hom))
        return_patch = True
    else:
        return_patch = False

    chex.assert_type(patch, int)
    z0, z = pop(hom, patch)

    if return_patch:
        return z / z0, patch.reshape(())
    else:
        return z / z0


@jax.jit
@auto_vmap(1, 0)
def affine_to_hom(affine: Array, patch: int) -> Array:
    """Convert affine coordinates to homogeneous coordinates.

    Args:
        affine: Affine coordinates.
        patch: Affine patch the coordinates are given in.
    Returns:
        Corresponding homogeneous coordinates.
    """
    chex.assert_type(patch, int)
    affine = jnp.atleast_1d(affine)
    one = jnp.ones(1, dtype=affine.dtype)
    return insert_1d(affine, one, patch)


def _fs_metric(z: Array, z_c: Array) -> Array:
    chex.assert_rank((z, z_c), 1)
    norm2p1 = 1 + jnp.sum(z * z_c)
    g = (jnp.eye(z.shape[0]) * norm2p1 - z_c.reshape(-1, 1) * z) / norm2p1 ** 2
    return g


@jax.jit
def fs_metric(zs: Array, zs_c: Array = None) -> Array:
    r"""Fubini-Study metric in terms of affine coordinates.

    This computes:

    .. math::
        \frac{1}{1+|z|^2} \mathbb{1} - \frac{1}{(1+|z|^2)^2} \bar{z} z^T

    Since the FS metric is symmetric, don't need the patch index as input.

    The coordinate array ``zs`` can have any shape (at least rank 1).
    The last dimension of the array is interpreted as the
    coordinate-index and gives the projective dimension.

    Args:
        zs: Affine coordinates.
        zs_c: (Optional) Complex conjugate affine coordinates.
            Can be explicitly passed to facilitate holomorphic gradient
            computation or to avoid re-computation.

    Returns:
        Matrix representing the value of the Fubini-Study metric
        and the given point. If the input is an array of coordinates,
        the output is a matching array of matrices.
    """
    (zs, zs_c), shape = util.flatten_coord(zs, zs_c, True)
    gs = jax.vmap(_fs_metric)(zs, zs_c)
    return util.unflatten_coord(gs, shape)


def _fs_potential(z: Array, z_c: Array) -> Array:
    chex.assert_rank((z, z_c), 1)
    return jnp.log(1 + jnp.sum(z * z_c))


def _fs_potential_hom(z: Array, z_c: Array) -> Array:
    chex.assert_rank((z, z_c), 1)
    return jnp.log(jnp.sum(z * z_c))


@partial(jax.jit, static_argnames=('hom',))
def fs_potential(zs: Array, zs_c: Array = None, hom: bool = False) -> Array:
    r"""Kaehler potential of FS metric.

    This computes:

    .. math::
        K = \ln(1 + |z|^2)

    Since the FS potential is symmetric, don't need the patch index as input.

    The coordinate array ``zs`` can have any shape (at least rank 1).
    The last dimension of the array is interpreted as the
    coordinate-index and determines the projective dimension.

    Args:
        zs: Affine coordinates.
        zs_c: (Optional) Complex conjugate affine coordinates.
            Can be explicitly passed to facilitate holomorphic gradient
            computation or to avoid re-computation.
        hom: If true, input is interpreted as homogeneous coordinates instead
            of affine coordinates.
    Returns:
        Real value representing the Fubini-Study potential at the given point.
    """
    (zs, zs_c), shape = util.flatten_coord(zs, zs_c, True)
    if hom:
        pot = jax.vmap(_fs_potential_hom)(zs, zs_c)
    else:
        pot = jax.vmap(_fs_potential)(zs, zs_c)
    return util.unflatten_coord(pot, shape)
