"""Methods based on the algebraic ansatz for the metric."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import numpy as np
import sympy

from .util import binomial, pop
from .differential import jacobian_embed
from .variety import VarietySingle
from .projective import hom_to_affine
from functools import partial
from chex import Array
from itertools import combinations_with_replacement
from progeval import ProgEval
from jax_autovmap import auto_vmap


def monomial_basis_indices(dim_proj: int, degree: int) -> Array:
    r"""Monomial basis as list of products.

    The output specifies indices :math:`I_{ij}` such that
    all monomials of degree ``self.degree`` on
    :math:`\mathbb{C}^{\mathrm{dim_proj}+1}` are given by

    .. math::
        s_i(z) = \prod_j z_{I_{ij}} \,.

    Returns:
        A 2d array where each row gives the index of the
        variable to be multiplied.
    """
    dim_complex = dim_proj + 1
    it = combinations_with_replacement(jnp.arange(dim_complex), degree)
    return jnp.array(list(it))


def monomial_basis(dim_proj: int, degree: int):
    r"""Return monomial basis as power matrix.

    The output specifies powers :math:`P_{ij}` such that
    all monomials of degree ``self.degree`` on
    :math:`\mathbb{C}^{\mathrm{dim_proj}+1}` are given by

    .. math::
        s_i(z) = \prod_j z_j^{P_{ij}} \,.

    Returns:
        A 2d array where each row gives the power of each of the coordinates.

    See Also: :py:func:`monomial_basis_indices`
    """
    if degree == 0:
        return jnp.zeros((1, dim_proj + 1), dtype=int)
    indices = monomial_basis_indices(dim_proj, degree)
    return jax.vmap(
        lambda idc: jax.lax.scan(
            lambda c, i: (c.at[i].add(1), None),
            jnp.zeros(dim_proj + 1, dtype=indices.dtype),
            idc
        )
    )(indices)[0]


def monomial_basis_size(dim_proj, degree, reduced=False):
    degree_def = dim_proj + 1
    basis = binomial(dim_proj + degree, degree)
    if not reduced or degree_def > degree:
        return basis
    mod = binomial(dim_proj + degree - degree_def, degree - degree_def)
    return basis - mod


@partial(jax.vmap, in_axes=(0, None))
def _getpows(zi, maxpow):
    last, zp = jax.lax.scan(lambda z, _: (z * zi, z), jnp.ones((), zi.dtype),
                            None, length=maxpow)
    return jnp.append(zp, last)


@partial(jax.jit, static_argnames=('maxdeg',))
@auto_vmap(zs=1, patch=0)
def _compute_monomials_index(zs, patch, pows, maxdeg):
    """Compute monomials in z, as defined by powers.
    Args:
        zs: Homogeneous coordinates of some dimension n.
        pows: N * n matrix of integers.
    """
    # pre-compute powers recursively to minimize cost
    zp = _getpows(zs, maxdeg)
    # transpose & remove patch coordinate, if given
    pows = pows.T if patch is None else pop(pows.T, patch)[1]
    zp = jax.vmap(lambda a, i: a[i])(zp, pows)
    return jnp.prod(zp, axis=0)


@jax.jit
@auto_vmap(zs=1, patch=0)
def _compute_monomials_log(zs, patch, pows):
    """Compute monomials in z, as defined by powers.
    Args:
        zs: Homogeneous coordinates of some dimension n.
        pows: N * n matrix of integers.
    """
    if patch is not None:
        pows = jnp.roll(pows, -patch, axis=1)[:, 1:]
        zs = jnp.roll(zs, -patch, axis=0)

    return jnp.exp(pows @ jnp.log(zs))


class LBSections:
    """General structure representing set of line bundle sections.

    The primary function of this class is that it can be called
    with coordinates and (optionally) an affine patch and return the values
    of the sections evaluated at those positions.

    This class itself is abstract and should be subclassed.
    Two concrete implementations are :class:`MonomialBasisFull`
    and :class:`MonomialBasisReduced`
    """
    #: Degree of sections.
    degree: int

    @property
    def size(self):
        """Number of sections."""
        raise NotImplementedError

    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, zs, patch, conj=False, **kwargs):
        """Evaluate line bundle sections at given coordinates.

        Args:
            zs: Affine coordinates.
            patch: Affine patch index.
            conj: If true, the output should compute the conjugate of the
                sections as an anti-holomorphic function.
                In this case, the ``zs`` input are complex conjugate
                coordinates. This argument is needed in case the definition of
                the sections contain complex parameters which need
                to be conjugated.
            **kwargs: Additional arguments.
        Returns:
            Array of evaluated
        """
        raise NotImplementedError


class _MonomialBasis(LBSections):
    """Building block for Monomial implementation of line bundle sections."""
    power_matrix: Array

    def __init__(self, degree: int, power_matrix: Array, ):
        super().__init__(degree)
        self.power_matrix = power_matrix

    @property
    def size(self):
        """Number of (basis) sections."""
        return len(self.power_matrix)

    def __call__(self, zs, patch, conj=False, precompute=True):
        if precompute:
            return _compute_monomials_index(
                zs, patch, self.power_matrix, self.degree)
        return _compute_monomials_log(zs, patch, self.power_matrix)


class MonomialBasisFull(_MonomialBasis):
    #: Matrix representing powers of monomials.
    power_matrix: Array
    #: Degree of monomials.
    degree: int

    def __init__(self, dim_proj: int, degree: int):
        """Full set of homogeneous monomials on projective space."""
        power_matrix = monomial_basis(dim_proj, degree)
        super().__init__(degree, power_matrix)


def reduced_monomial_basis(degree, poly_pows):
    r"""Generate power matrix for monomial basis modulo a polynomial.

    Denote the full set of degree :math:`d_s` (=``degree``)
    monomials on :math:`\mathbb{C}^{\mathrm{dim_proj}+1}` as :math:`s_i(z)`.

    On the set where the degree :math:`d_p` homogeneous polynomial
    :math:`p(z) = 0`, the monomials :math:`s_i(z)` become linearly dependent.
    This function implements a simple, although possibly not optimal,
    scheme to obtain a reduced basis of monomials on that set.

    If the monomial degree is less than the degree of the polynomial,
    nothing has to be done.

    Where the polynomial :math:`p(z)` vanishes, any linear combination
    of basis monomials proportional to :math:`p` vanishes.
    Let :math:`r_j(z)` denote the degree :math:`d_s - d_p` monomials.
    The linear dependencies introduced by :math:`p(z)=0` are then

    .. math::
        r_j(z) p(z) = 0 \,,

    which is a sum of degree :math:`d_s` monomials.
    This function produces a basis by removing one of these
    (linear dependent) monomials for each  :math:`r_j`
    from the full set of monomials :math:`s_j`.


    Args:
        poly_pows: 2d integer matrix, giving the monomials which make up the
            polynomial to be modded out from basis.
    Returns:
        2d integer matrix.
    """
    poly_pows = jnp.atleast_2d(poly_pows)
    mod_degree = np.sum(poly_pows, 1)
    assert np.all(mod_degree == np.roll(mod_degree, 1))
    mod_degree = mod_degree[0]

    # number of homogeneous coordinates given by last index
    dim_proj = poly_pows.shape[1] - 1

    if degree < mod_degree:
        return monomial_basis(dim_proj, degree)

    pows_aux = monomial_basis(dim_proj, degree - mod_degree)
    pows = [tuple(p) for p in monomial_basis(dim_proj, degree)]

    for pow_aux in pows_aux:
        # need to remove one component (i.e. row) of these to remove
        # the element from the span of the pows-basis
        mod_element = poly_pows + pow_aux.reshape(1, -1)
        for mon in mod_element:
            try:
                pows.remove(tuple(mon))
                break  # done if removed one
            except ValueError:
                continue  # already removed this one, remove another
        else:
            # If break was never called in for loop, couldn't reduce basis
            raise RuntimeError('Basis reduction scheme failed.')

    return jnp.array(pows)


class MonomialBasisReduced(_MonomialBasis):
    #: Matrix representing powers of monomials.
    power_matrix: Array
    #: Degree of monomials.
    degree: int

    def __init__(self, variety: VarietySingle, degree: int, params):
        """Reduced set of homogeneous monomials giving basis on variety.

        Note that procedure used to reduce the basis here may not
        be numerically optimal.

        See Also: :func:`reduced_monomial_basis`

        Args:
            variety: Variety for which to find reduced basis.
            degree: Degree of monomials.
            params: Complex moduli parameters for variety.
                The reduction algorithm depends on the form of the defining
                polynomial and is thus in general dependent on moduli values.
        """
        zs = sympy.symarray('z', variety.dim_projective + 1)
        poly = variety.defining_poly(zs, np.array(params))
        pows_mod = np.array(
            [t[0] for t in sympy.Poly(poly, domain='CC').terms()])
        power_matrix = reduced_monomial_basis(degree, pows_mod)
        super().__init__(degree, power_matrix)


def _transform_if_static(function, static, _):
    """Transform the function if it is static."""
    doc = function.__doc__
    if doc is not None and doc.startswith('@math'):
        # make math-docstrings one line
        doc = ':math:`' + doc[5:].replace('\n', '').strip() + '`'
        function.__doc__ = doc
    return jax.jit(function, inline=True) if static else function


class GeometricObjects(ProgEval, transformer=_transform_if_static):
    #: Affine coordinates.
    zs: jnp.ndarray
    #: Affine patch index.
    patch: int
    #: Values of line bundle sections :math:`s_\alpha(z)`
    s: jnp.ndarray
    #: :math:`[\partial_i s_\alpha(z)]_{\alpha, i}`
    s_1: jnp.ndarray
    #: :math:`[\partial_i \partial_{\bar{\jmath}} s_\alpha(z)]_{\alpha, i, \jmath}`
    s_2: jnp.ndarray

    def __init__(self,
                 h: jnp.ndarray, zs: jnp.ndarray, params: jnp.ndarray,
                 variety: VarietySingle, sections: LBSections,
                 patch: int = None,
                 grad_def: jnp.ndarray = None,
                 dependent: int = None,
                 **section_args):
        """Computational graph for objects derived from algebraic ansatz.

        To avoid duplicate code for functions computing different
        geometric objects derived from the algebraic ansatz, this computational
        graph collects their dependencies together with intermediate variables.
        The graph is evaluated lazily based on property access.

        Note that this object takes single values as input (i.e. without
        batch dimension).

        Args:
            h: Hermitian matrix in algebraic ansatz.
            zs: Coordinates on the variety, either homogeneous or affine
                (if a patch index is provided).
            params: Complex moduli parameters of the variety.
            variety: Manifold for which
            sections: Set of sections on the line bundle which define the
                embedding into higher order projective spaces.
            patch: Optional affine patch index. If ``None``, ``zs``
                are assumed to be homogeneous coordinates.
            grad_def: Optional gradient of the defining polynomial, to save
                re-computation.
            dependent: Optional index of the dependent coordinate.
                This index is with respect to the affine coordinate array.
            **section_args: Additional keyword arguments are passed to
                the sections object.
        """
        super().__init__(
            h=h, params=params, degree=sections.degree, variety=variety)

        if patch is None:
            zs, patch = hom_to_affine(zs)

        self.zs = zs
        self.patch = patch

        assert zs.shape[-1] == variety.dim_projective, \
            'coordinates do not match variety dimension'

        def pow_fn(zs):
            return sections(zs, patch, **section_args)

        jac_fn = jax.jacfwd(pow_fn, holomorphic=True)

        # compute sections & gradients
        self.s = pow_fn  # computes monomials
        self.register('s_1', jac_fn, ['zs'])
        self.register('s_2', jax.jacfwd(jac_fn, holomorphic=True), ['zs'])

        if grad_def is not None:
            self.grad_def = grad_def
        if dependent is not None:
            self.dependent = dependent

    @staticmethod
    def dependent(grad_def):
        r"""Dependent coordinate index :math:`\delta`.

        Omitting this index from the affine coordinates defines local
        coordinates on the variety.
        Given these, the omitted value can be recovered using the defining
        equation.
        """
        return jnp.argmax(jnp.abs(grad_def))

    # compute sections & gradients

    @staticmethod
    def s_c(s):
        r"""@math \bar{s}_{\bar{\alpha}}(\bar{z})"""
        return jnp.conj(s)

    @staticmethod
    def s_1_c(s_1):
        r"""@math
        [
            \partial_\bar{\imath}
            \bar{s}_{\bar{\alpha}}(\bar{z})
        ]_{\bar{\alpha},\bar{\imath}}
        """
        return jnp.conj(s_1)

    @staticmethod
    def s_2_c(s_2):
        r"""@math
        [
            \partial_\bar{\imath} \partial_{\bar{\jmath}}
            \bar{s}_{\bar{\alpha}}(\bar{z})
        ]_{\bar{\alpha}, \bar{\imath}, \bar{\jmath}}
        """
        return jnp.conj(s_2)

    # compute contractions of sections with h & gradients
    # _h and _a denote holomorphic and antiholomorphic gradients

    @staticmethod
    def kahler(psi, degree):
        r"""Kahler potential :math:`K = \log(\psi)/\pi k`."""
        return jnp.log(psi) / (jnp.pi * degree)

    @staticmethod
    def psi(h, s, s_c):
        r"""@math \psi=s_\alpha H^{\alpha\bar{\beta}} \bar{s}_{\bar{beta}}"""
        if h.ndim == 1:
            return jnp.sum(s * h * s_c)
        return jnp.einsum('i,ij,j', s, h, s_c)

    @staticmethod
    def psi_h(h, s_1, s_c):
        r"""@math \partial_i \psi"""
        if h.ndim == 1:
            return jnp.einsum('ia,i,i->a', s_1, h, s_c)
        return jnp.einsum('ia,ij,j->a', s_1, h, s_c)

    @staticmethod
    def psi_a(psi_h):
        r"""@math \partial_{\bar{\imath}} \psi"""
        return jnp.conj(psi_h)

    @staticmethod
    def psi_ha(h, s_1, s_1_c):
        r"""@math \partial_{i} \partial_{\bar{\jmath}} \psi"""
        if h.ndim == 1:
            return jnp.einsum('ia,i,ib->ab', s_1, h, s_1_c)
        return jnp.einsum('ia,ij,jb->ab', s_1, h, s_1_c)

    @staticmethod
    def psi_hh(h, s_2, s_c):
        r"""@math \partial_{i} \partial_{j} \psi"""
        if h.ndim == 1:
            jnp.einsum('iab,i,i->ab', s_2, h, s_c)
        return jnp.einsum('iab,ij,j->ab', s_2, h, s_c)

    @staticmethod
    def psi_aa(psi_hh):
        r"""@math \partial_{\bar{\imath}} \partial_{\bar{\jmath}} \psi"""
        return jnp.conj(psi_hh)

    @staticmethod
    def psi_hha(h, s_2, s_1_c):
        r"""@math \partial_i \partial_j \partial_{\bar{k}} \psi"""
        if h.ndim == 1:
            return jnp.einsum('iab,i,ic->abc', s_2, h, s_1_c)
        return jnp.einsum('iab,ij,jc->abc', s_2, h, s_1_c)

    @staticmethod
    def psi_hhaa(h, s_2, s_2_c):
        r"""@math \partial_i \partial_j \partial_{\bar{k}} \partial_{\bar{l}} \psi"""
        if h.ndim == 1:
            return jnp.einsum('iab,i,icd->abcd', s_2, h, s_2_c)
        return jnp.einsum('iab,ij,jcd->abcd', s_2, h, s_2_c)

    # compute metric in ambient coordinates (to be pulled back)
    # and gradients of it

    @staticmethod
    def g_proj(psi, psi_h, psi_a, psi_ha, degree):
        r"""@math \hat{g}"""
        d = degree * jnp.pi
        return (psi_ha / psi - psi_h[:, None] * psi_a[None, :] / psi ** 2) / d

    @staticmethod
    def psi_h_psi_ha(psi_h, psi_ha):
        r"""@math \tilde{\Psi}^{(2,3)}_{ij\bar{k}} = \psi_{i} \psi_{j\bar{k}}"""
        return jnp.einsum('a,ij->aij', psi_h, psi_ha)

    @staticmethod
    def two_hha(psi_a, psi_hh, psi_h_psi_ha):
        r"""@math
        \Psi^{(2,3)}_{ij\bar{k}}
            = \psi_{i} \psi_{j\bar{k}}
            + \psi_{j} \psi_{i\bar{k}}+\psi_{ij} \psi_{\bar{k}}
        """
        psi_hh_psi_a = jnp.einsum('j,ai->aij', psi_a, psi_hh)
        return psi_h_psi_ha + psi_h_psi_ha.swapaxes(0, 1) + psi_hh_psi_a

    @staticmethod
    def three_hha(psi_h, psi_a):
        r"""@math
        \Psi^{(3,3)}_{ij\bar{k}}
            = \psi_{i} \psi_{j} \psi_{\bar{k}}
        """
        return jnp.einsum('a,i,j->aij', psi_h, psi_h, psi_a)

    @staticmethod
    def g_proj_h(psi, psi_hha, two_hha, three_hha, degree):
        r"""@math
        \partial_a \hat{g}_{i\bar{\jmath}} =
            \frac{1}{\pi k} \left(
                \psi_{ai\bar{\jmath}}/\psi +
                \Psi^{(2,3)}_{ai\bar{\jmath}}/\psi^2 +
                2\Psi^{(3,3)}_{ai\bar{\jmath}}/\psi^3
            \right)
        """
        d = degree * jnp.pi
        return (psi_hha / psi - two_hha / psi**2 + three_hha * 2 / psi**3) / d

    @staticmethod
    def g_proj_a(g_proj_h):
        r"""@math \partial_{\bar{a}} \hat{g}_{i\bar{\jmath}}"""
        return jnp.conj(g_proj_h).transpose((0, 2, 1))

    @staticmethod
    def psi_hha_psi_a(psi_a, psi_hha):
        r"""@math
        \tilde{\Psi}^{(2,4)}_{i\bar{\jmath}k\bar{l}}
            = \psi_{ik\bar{l}} \psi_{\bar{\jmath}}
        """
        return jnp.einsum('aij,b->abij', psi_hha, psi_a)

    @staticmethod
    def psi_ha_psi_ha(psi_ha):
        r"""@math
        \left[\psi_{i\bar{\jmath}} \psi_{k\bar{l}}\right]_{ij\bar{k}\bar{l}}
        """
        return jnp.einsum('ab,ij->abij', psi_ha, psi_ha)

    @staticmethod
    def two_haha(psi_hh, psi_aa, psi_ha_psi_ha, psi_hha_psi_a):
        r"""@math
        \Psi^{(2,4)}_{i\bar{\jmath}k\bar{l}}
            = \psi_{ik\bar{l}} \psi_{\bar{\jmath}}
            + \psi_{k\bar{l}\bar{\jmath}} \psi_{i}
            + \psi_k \psi_{i\bar{\jmath}\bar{l}}
            + \psi_{\bar{l}} \psi_{ik\bar{\jmath}}
            + \psi_{i\bar{\jmath}} \psi_{k\bar{l}}
            + \psi_{k\bar{\jmath}} \psi_{i\bar{l}}
            + \psi_{\bar{\jmath}\bar{l}}\psi_{ik}
        """
        return (psi_hha_psi_a + psi_hha_psi_a.swapaxes(1, 3) +
                jnp.conj(psi_hha_psi_a).transpose((1, 2, 3, 0)) +
                jnp.conj(psi_hha_psi_a).transpose((3, 2, 1, 0)) +
                psi_ha_psi_ha + psi_ha_psi_ha.swapaxes(0, 2) +
                jnp.einsum('bj,ai->abij', psi_aa, psi_hh))

    @staticmethod
    def three_haha(psi_h, psi_a, psi_aa, two_hha, psi_h_psi_ha):
        r"""@math
        \Psi^{(3,4)}_{i\bar{\jmath}k\bar{l}}
            = \Psi^{(2,3)}_{ik\bar{l}} \psi_{\bar{\jmath}}
            + \left(\tilde{\Psi}^{(2,3)}_{ki\bar{\jmath}}
            + \tilde{\Psi}^{(2,3)}_{ik\bar{\jmath}}\right) \psi_{\bar{l}}
            + \psi_i \psi_k \psi_{\bar{\jmath}\bar{l}}
        """
        return (jnp.einsum('aij,b->abij', two_hha, psi_a) +
                jnp.einsum('iab,j->abij',
                           psi_h_psi_ha + psi_h_psi_ha.swapaxes(0, 1), psi_a) +
                jnp.einsum('a,i,bj->abij', psi_h, psi_h, psi_aa))

    @staticmethod
    def four_haha(psi_a, three_hha):
        r"""@math
        \Psi^{(4,4)}_{i\bar{\jmath}k\bar{l}}
            = \Psi^{(3,3)}_{ik\bar{l}} \psi_{\bar{\jmath}}
        """
        return jnp.einsum('aij,b->abij', three_hha, psi_a)

    @staticmethod
    def g_proj_ha(psi, psi_hhaa, two_haha, three_haha, four_haha, degree):
        r"""@math
        \partial_a \partial_{\bar{b}} \hat{g}_{i\bar{\jmath}}
            = \frac{1}{\pi k} \left(
                \psi_{ai\bar{b}\bar{\jmath}}/\psi
                - \Psi^{(2,3)}_{a\bar{b}i\bar{\jmath}}/\psi^2
                + 2 \Psi^{(3,4)}_{a\bar{b}i\bar{\jmath}}/\psi^3
                - 6\Psi^{(4,4)}_{a\bar{b}i\bar{\jmath}}/\psi^4
            \right)
        """
        d = degree * jnp.pi
        return (psi_hhaa.transpose((0, 2, 1, 3)) / psi - two_haha / psi ** 2
                + three_haha * 2 / psi ** 3 - four_haha * 6 / psi ** 4) / d

    # compute the Jacobian of the embedding into ambient projective space

    def grad_def(self, zs, params, patch):
        r"""@math G_i=\partial_i Q"""
        return self.variety.grad_defining(zs, params, patch)

    @staticmethod
    def jac(grad_def, dependent):
        r"""@math J_{ij}=\delta_{ij} - \delta_{j\delta} \, G_i/ G_\delta"""
        return jacobian_embed(grad_def, dependent)

    @staticmethod
    def jac_c(jac):
        r"""@math \bar{J}_{ij}"""
        return jnp.conj(jac)

    # compute induced metric & ricci tensor

    def jac_h(self, zs, params, patch, dependent):
        r"""@math J_{ij,a}=\partial_a J_{ij}"""
        return jax.jacfwd(self.variety.jacobian_embed, 0, holomorphic=True)(
            zs, params, patch, dependent)

    @staticmethod
    def g_loc(g_proj, jac, jac_c):
        r"""@math g=J\hat{g}J^\dagger"""
        return jnp.einsum('ri,ij,sj->rs', jac, g_proj, jac_c)

    @staticmethod
    def g_loc_h(g_proj, g_proj_h, jac, jac_h, jac_c):
        r"""@math
        \partial_a g
            = (\partial_a J) \hat{g} J^\dagger
            + J \partial_a \hat{g} J^\dagger
        """
        return (jnp.einsum('ria,ij,sj->ars', jac_h, g_proj, jac_c) +
                jnp.einsum('ri,aij,sj->ars', jac, g_proj_h, jac_c))

    @staticmethod
    def g_loc_ha(g_proj, g_proj_a, g_proj_h, g_proj_ha, jac, jac_c, jac_h):
        r"""@math
        \partial_a \partial_{\bar{b}} g
            = \partial_a J \partial_{\bar{b}}\hat{g}J^\dagger
            + J\partial_a\partial_{\bar{b}}\hat{g}J^\dagger
            + \partial_a J \hat{g} \partial_{\bar{b}}J^\dagger
            + J \partial_a \hat{g}\partial_{\bar{b}}J^\dagger
        """
        jac_c_a = jnp.conj(jac_h)
        return (jnp.einsum('ria,bij,sj->abrs', jac_h, g_proj_a, jac_c) +
                jnp.einsum('ri,abij,sj->abrs', jac, g_proj_ha, jac_c) +
                jnp.einsum('ria,ij,sjb->abrs', jac_h, g_proj, jac_c_a) +
                jnp.einsum('ri,aij,sjb->abrs', jac, g_proj_h, jac_c_a))

    @staticmethod
    def g_loc_inv(g_loc):
        r"""@math g^{-1}"""
        return jnp.linalg.inv(g_loc)

    @staticmethod
    def g_loc_a(g_loc_h):
        r"""@math \partial_{\bar{a}}g^\dagger"""
        return jnp.conj(g_loc_h).swapaxes(2, 1)

    @staticmethod
    def ricci_proj(g_loc_h, g_loc_a, g_loc_ha, g_loc_inv):
        r"""@math
        \hat{R}_{a\bar{b}}
            = \mathrm{tr}[g^{-1} \partial_a \partial_{\bar{b}} g]
            - \mathrm{tr}[ g^{-1} (\partial_a g) g^{-1} \partial_{\bar{b}} g]
        """
        # here R = - d d log det g
        return jnp.einsum(
            'ji,aik,kl,blj', g_loc_inv, g_loc_h, g_loc_inv, g_loc_a) - \
            jnp.einsum('ji,abij->ab', g_loc_inv, g_loc_ha)

    @staticmethod
    def ricci_loc(ricci_proj, jac, jac_c):
        r"""@math R = J\hat{R} J^\dagger"""
        return jnp.einsum('ri,ij,sj->rs', jac, ricci_proj, jac_c)

    @staticmethod
    def ricci_scalar(ricci_loc, g_loc_inv):
        r"""@math \mathrm{tr}\, R g^{-1}"""
        return jnp.einsum('ij,ji', ricci_loc, g_loc_inv)

    @staticmethod
    def eta(g_loc, grad_def, dependent):
        r"""@math \eta = |G_\delta|^2 \det g"""
        return jnp.linalg.det(g_loc) * jnp.abs(grad_def[dependent]) ** 2


class AlgebraicMetric:
    #: Complex projective variety.
    variety: VarietySingle
    #: Choice of line bundle sections.
    sections: LBSections

    @property
    def degree(self) -> int:
        """Degree of algebraic variety."""
        return self.sections.degree

    def __init__(self, variety: VarietySingle, sections: LBSections):
        """Algebraic metric object.

        This wraps the variety and linen bundle sections objects
        and provides functions to compute geometric objects based on
        the algebraic ansatz.

        All functions can be jit-compiled and automatically handle
        any combination of batch dimensions for the inputs.
        """
        self.sections = sections
        self.variety = variety

    @auto_vmap(zs=1, patch=0, params=1, h=2, dependent=0, grad_def=1)
    def kahler_potential(self, h, zs, params, patch=None, dependent=None, grad_def=None):
        """Compute the Kahler potential.

        Args:
            h: Hermitian matrix.
            zs: affine or homogeneous coordinates.
            params: Complex moduli parameters of variety.
            patch: Affine patch index for affine coordinates.
            dependent: Optional, dependent coordinate index.
            grad_def: Optional, gradient of defining polynomial.

        Returns:
            Kahler potential evaluated at given point(s).
        """
        obj = GeometricObjects(
            h, zs, params,
            self.variety, self.sections,
            patch, grad_def, dependent)
        return obj.kahler

    @auto_vmap(zs=1, patch=0, params=1, h=2, dependent=0, grad_def=1)
    def metric(self, h, zs, params, patch=None, dependent=None, grad_def=None):
        """Compute the metric in terms of local coordinates.

        Args:
            h: Hermitian matrix.
            zs: affine or homogeneous coordinates.
            params: Complex moduli parameters of variety.
            patch: Affine patch index for affine coordinates.
            dependent: Optional, dependent coordinate index.
            grad_def: Optional, gradient of defining polynomial.

        Returns:
            Metric, affine patch index, dependent coordinate index.
        """
        obj = GeometricObjects(
            h, zs, params,
            self.variety, self.sections,
            patch, grad_def, dependent)
        return obj.g_loc, obj.patch, obj.dependent

    @auto_vmap(zs=1, patch=0, params=1, h=2, dependent=0, grad_def=1)
    def eta(self, h, zs, params, patch=None, dependent=None, grad_def=None):
        """Compute the eta ratio.

        Args:
            h: Hermitian matrix.
            zs: affine or homogeneous coordinates.
            params: Complex moduli parameters of variety.
            patch: Affine patch index for affine coordinates.
            dependent: Optional, dependent coordinate index.
            grad_def: Optional, gradient of defining polynomial.

        Returns:
            The eta ratio evaluated at given point(s).
        """
        obj = GeometricObjects(
            h, zs, params,
            self.variety, self.sections,
            patch, grad_def, dependent)
        return obj.eta

    @auto_vmap(zs=1, patch=0, params=1, h=2, dependent=0, grad_def=1)
    def ricci(self, h, zs, params, patch=None, dependent=None, grad_def=None):
        """Compute the Ricci curvature tensor in local coordinates.

        Args:
            h: Hermitian matrix.
            zs: affine or homogeneous coordinates.
            params: Complex moduli parameters of variety.
            patch: Affine patch index for affine coordinates.
            dependent: Optional, dependent coordinate index.
            grad_def: Optional, gradient of defining polynomial.

        Returns:
            Ricci curvature tensor,
            affine patch index,
            dependent coordinate index.
        """
        obj = GeometricObjects(
            h, zs, params,
            self.variety, self.sections,
            patch, grad_def, dependent)
        return obj.ricci_loc, obj.patch, obj.dependent

    @auto_vmap(zs=1, patch=0, params=1, h=2, dependent=0, grad_def=1)
    def ricci_scalar(self, h, zs, params, patch=None, dependent=None, grad_def=None):
        """Compute the Kahler potential.

        Args:
            h: Hermitian matrix.
            zs: affine or homogeneous coordinates.
            params: Complex moduli parameters of variety.
            patch: Affine patch index for affine coordinates.
            dependent: Optional, dependent coordinate index.
            grad_def: Optional, gradient of defining polynomial.

        Returns:
            Ricci scalar evaluated at given point(s).
        """
        obj = GeometricObjects(
            h, zs, params,
            self.variety, self.sections,
            patch, grad_def, dependent)
        return obj.ricci_scalar

    @auto_vmap(params=1, h=2)
    def sigma_accuracy(self, key, params, h, count):
        r"""The :math:`\sigma` accuracy measure.

        The :math:`\sigma` measure is the integral
        of :math:`|1-\eta|` over the manifold with respect to the
        holomorphic volume form.

        Args:
            key: Random key for point-sampling used in MC integration.
            h: Hermitian matrix.
            params: Complex moduli parameters of variety.
            count: Number of points used in Monte Carlo approximation
                of the integral.
        Returns:
            The :math:`sigma` accuracy measure.
        """
        zs, patch = self.variety.sample_intersect(key, params, count, affine=True)

        grad_def = self.variety.grad_defining(zs, params, patch)
        dep = jnp.argmax(jnp.abs(grad_def), axis=1)

        weights = self.variety.sample_intersect_weights(zs, params, patch, dep)

        etas = self.eta(h, zs, params, patch, dep, grad_def)
        vol_k = jnp.mean(etas * weights)  # (dVolK / dVol * dVol / dA) dA
        vol_cy = jnp.mean(weights).real

        sigma = jnp.mean(jnp.abs(1 - etas * vol_cy / vol_k) * weights) / vol_cy
        return sigma

    def donaldson_step(self, key, h, params, vol_cy, batches, batch_size):
        """Single step of Donaldson's algorithm.

        Args:
            key: Random key for point-sampling used in MC integration.
            h: Hermitian matrix.
            params: Complex moduli parameters of variety.
            vol_cy: Volume of the variety using the canonical volume form.
            batches: Number of batches used in the Monte Carlo integration.
            batch_size: Number of points sampled in each batch of the
                Monte Carlo integration.

        Returns:
            New value of the H matrix.
        """
        T = jnp.zeros_like(h)
        h = (h + h.T.conj()) / 2
        keys = jax.random.split(key, batches)

        def batch_step(i, T):
            key_sample = keys[i]
            (zs, patch), weights = self.variety.sample_intersect(key_sample, params, batch_size, affine=True,
                                                                 weights=True)
            weights = jnp.nan_to_num(weights, posinf=0, neginf=0, nan=0)
            # seems to be cheaper as long as not taking derivatives
            mon = self.sections(zs, patch)
            mon_c = mon.conj()

            num = jnp.einsum('ia,ib->iab', mon, mon_c)
            den = jnp.einsum('ia,ib,ab->i', mon, mon_c, h).real
            delta_T = jnp.mean(num * (weights / den).reshape(-1, 1, 1), axis=0)
            return T + delta_T / batches

        T = jax.lax.fori_loop(0, batches, batch_step, T)

        T *= h.shape[0] / vol_cy
        new_h = jnp.linalg.inv(T).T
        return new_h
