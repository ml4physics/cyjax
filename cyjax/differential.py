from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import update_wrapper
from . import util
from .util import Array
from typing import Callable
from jax_autovmap import auto_vmap


def complex_hessian(
        potential: Callable[..., Array],
        z_index: int = 0, z_c_index: int = 1) -> Callable[..., Array]:
    r"""Return the mixed hessian of a function.

    Given a potential function :math:`f` which takes both the coordinates
    and the complex conjugate as input (so we can compute the holomorphic
    derivatives), this generates a new function which computes

    .. math::
        (z, \bar{z}, \ldots) \mapsto \partial_i \bar{\partial}_j f(z, \bar{z}, \ldots)

    Args:
        potential: Potential function which takes affine coordinates and their
            conjugate as input. May also take other arguments as input;
            the order of input arguments is preserved. The output must be
            a real value.
        z_index: Argument index of affine coordinate in potential function.
        z_c_index: Argument index of complex conjugate affine coordinates
            in potential function.

    Returns:
        A function taking affine coordinate and complex conjugate coordinate
        as input and returns the complex Hessian matrix.
    """
    hessian = jax.jacfwd(
        jax.jacrev(potential, z_index, holomorphic=True),
        z_c_index, holomorphic=True)

    _docs = potential.__doc__ if hasattr(potential, '__doc__') else ''
    if hasattr(potential, '__name__'):
        docs = f'Mixed hessian of {potential.__name__}\n\n{_docs}'
        name = 'hessian_of_' + potential.__name__
    else:
        name = 'hessian'
        docs = f'Mixed hessian\n\n{_docs}'
    fun = update_wrapper(hessian, potential)
    fun.__name__ = name
    fun.__doc__ = docs
    return fun


@jax.jit
@auto_vmap(grad_def=1, dependent=0)
def jacobian_embed(grad_def: Array, dependent: int) -> Array:
    r"""Jacobian of the defining embedding into ambient projective space.

    Let :math:`X` be a variety embedded in the ambient projective space
    :math:`\mathbb{P}^{d+1}` as the zero-locus of a homogeneous polynomial
    :math:`Q(z)=0`, where :math:`z` are ambient affine coordinates.
    Denote by :math:`x` local coordinates on the variety.
    Explicitly, the embedding would be given by :math:`z(x)`.
    However, we can compute the Jacobian :math:`\partial z / \partial x`
    needed for pullbacks without explicitly deriving this embedding.

    Here, let :math:`x` be the local coordinates obtained by omitting
    :math:`z[\mathrm{dep}]`.
    The output of this function is the matrix :math:`(dz_i/dx_j)_{ij}`
    computed by using

    .. math::
        \frac{dz_{\mathrm{dep}}}{dx_j} =
        - \frac{dQ}{dz_j} \left( \frac{Q}{dz_{\mathrm{dep}}} \right)^{-1} \,.

    Args:
        grad_def: Derivative of defining equation with respect to
            affine coordinates; Array of length ``d``.
        dependent: Index of dependent coordinate in
            affine coordinate vector.
    Returns:
        :math:`(d-1 \times d)` array.
        The Jacobian for the defining embedding.
    """
    jac = jnp.eye(grad_def.size - 1, dtype=grad_def.dtype)
    grad_def_dep, grad_def_indep = util.pop(grad_def, dependent)
    col_dep = - grad_def_indep / grad_def_dep
    return util.insert_col(jac, col_dep, dependent)


@jax.jit
@auto_vmap(metric=2, grad_def=1, dependent=0, grad_def_c=1)
def induced_metric(
        metric: Array, grad_def: Array, dependent: int,
        grad_def_c: Array = None) -> Array:
    r"""Induced metric in local patch coordinates of variety.

    If :math:`x` are local coordinates and the :math:`z` corresponding
    affine coordinates in ambient space, this function computes the
    pullback of a metric :math:`g_{ij}` to the variety given the gradients
    :math:`\partial P / \partial z_i` of its defining equation:

    .. math::
        g^X_{k\bar{l}} =
            \frac{\partial z^{(p)}_{i}}{\partial z^{(p)}_{k}}
            \frac{\partial \bar{z}^{(p)}_{\bar{\jmath}}}
                 {\partial \bar{z}^{(p)}_{\bar{l}}} g_{i\bar{\jmath}}

    With the ambient projective space being :math:`d+1` dimensional,
    the metric :math:`g_{ij}` in affine coordinates has indices :math:`i, j`
    ranging from `0` to :math:`d`.
    Local coordinates on the variety are defined by omitting index
    ``local_dep_index`` from the affine coordinate vector.
    The pullback of the metric will thus have one fewer value in each
    index :math:`k, l`.

    Args:
        metric: Two-dimensional :math:`(d+1) \times (d+1)` array
            specifying the metric in ambient affine coordinates.
        dependent: Index of the dependent coordinate in the
            affine coordinate vector.
        grad_def: Array :math:`dP / dz_i` where :math:`P(z) = 0` is the
            defining polynomial.
        grad_def_c: Optionally pass the complex conjugate of the gradient
            of the defining equation to avoid re-computation.

    Returns:
        A :math:`d \times d` matrix;
        pullback of the ambient metric to local coordinates.

    See Also: :func:`jacobian_embed`
    """
    jac = jacobian_embed(grad_def, dependent)
    if grad_def_c is not None:
        jac_c = jacobian_embed(grad_def_c, dependent)
    else:
        jac_c = jac.conj()

    return jnp.einsum('ij,kl,jl->ik', jac, jac_c, metric)
