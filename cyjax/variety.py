from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

import sympy
from math import factorial

from . import util
from .differential import jacobian_embed, induced_metric
from .random import complex_sphere
from .projective import fs_metric, hom_to_affine, index_affine_to_hom
from .polynomial import HomPoly, FermatPoly, DworkPoly, univariate_coefficients, Poly
from .util import mc_integrate_batched as _int
from typing import Union, Sequence
from functools import partial
from jax_autovmap import auto_vmap


@partial(jax.vmap, in_axes=(0, 0, None, None))
def _solve_intersect(ps, qs, params, coeffs):
    """Find points z where z = q + t * p and Q(z, psi) = 0."""
    # p and q are in homogeneous coordinates
    coeffs = jnp.array([
            coeff(ps, qs, params) if isinstance(coeff, Poly) else coeff
            for coeff in coeffs])
    roots = jnp.roots(coeffs, strip_zeros=False)
    return ps.reshape(1, -1) + roots.reshape(-1, 1) * qs.reshape(1, -1)


class VarietySingle:
    _hom_poly: HomPoly

    def __init__(self, hom_poly: HomPoly):
        """Complex projective variety given by single defining equation.

        Instead of using a :class:`HomPoly` object, varieties can also be
        created using the classmethod :func:`VarietySingle.from_sympy`.
        """
        assert len(hom_poly.variable_names) == 1, \
            f'{type(self).__name__} only supports defining polynomials ' \
            f'over a single n-dimensional complex projective space.'
        self._hom_poly = hom_poly
        self._defining = hom_poly.transform_eval()

    @classmethod
    def from_sympy(cls,
                   poly: Union[str, sympy.Expr, sympy.Poly],
                   var_name: str = 'z', poly_degree: int = 5,
                   params: Sequence[Union[sympy.Symbol, str]] = None):
        """Construct the variety object from the defining equation.

        Args:
            poly: Defining homogeneous polynomial.
            var_name: Name of the variable polynomial is defined in.
            poly_degree: Degree of the defining equation.
            params: List of parameters in the defining equation.
                These can be automatically collected as they appear in
                the expression for the polynomial.

        Returns:
            A new :class:`HomPoly` object.
        """
        poly = HomPoly.from_sympy(poly, [var_name], params, [poly_degree])
        return cls(poly)

    @property
    def num_defining(self) -> int:
        """Number of coordinates fixed by defining equation."""
        return 1

    @property
    def parameters(self) -> Sequence[sympy.Symbol]:
        """List the symbols which parametrize the defining equation."""
        return self._hom_poly.parameters

    @property
    def par_count(self) -> int:
        """Number of parameters in defining equation."""
        return len(self.parameters)

    @property
    def dim_complex(self) -> int:
        """Dimension of ambient complex space."""
        return len(self._hom_poly.variable_indices[0])

    @property
    def dim_projective(self) -> int:
        """Dimension of ambient projective space."""
        return self.dim_complex - 1

    @property
    def dim_variety(self) -> int:
        """Dimension of variety."""
        return self.dim_projective - 1

    @property
    def dim(self) -> int:
        """Dimension of variety."""
        return self.dim_variety

    @property
    def defining_poly(self) -> HomPoly:
        """The defining homogeneous polynomial."""
        return self._hom_poly

    def defining(self,
                 zs: jnp.ndarray,
                 params: jnp.ndarray = None,
                 patch: Union[jnp.ndarray, int] = None,
                 **kwargs):
        """Defining equation of the variety as embedded in ambient space.

        This function works for numerical inputs. For symbolic inputs,
        see :attr:`VarietySingle.defining_poly`.
        """
        return self._defining(zs, params=params, patch=patch, **kwargs)

    @auto_vmap(zs=1, params=1, patch=0)
    def grad_defining(self, zs, params, patch=None):
        """Holomorphic gradient of defining equation."""
        return jax.grad(self.defining, 0, holomorphic=True)(zs, params, patch)

    @auto_vmap(zs=1, params=1, patch=0)
    def best_dependent(self, zs, params, patch):
        """Compute numerically optimal dependent coordinate.

        It is determined as the index for which the gradient (of the
        defining equation) has the largest absolute value.
        This is numerically optimal, since we divide by this value later,
        and thus avoid large values.
        """
        dependent = jnp.argmax(
            jnp.abs(self.grad_defining(zs, patch, params)), axis=1)
        return dependent

    @auto_vmap(zs=1, params=1, patch=0, dependent=0)
    def jacobian_embed(self, zs, params, patch, dependent=None):
        """Jacobian of embedding into ambient projective space.

        If dependent is None, also return chosen dependent index.
        """
        grad_def = self.grad_defining(zs, params, patch)

        if dependent is None:
            dependent = jnp.argmax(jnp.abs(grad_def))
            return jacobian_embed(grad_def, dependent), dependent
        return jacobian_embed(grad_def, dependent)

    def induced_metric(self, metric, zs, params, patch,
                       zs_c=None, dependent=None):
        """Induced metric given ambient CPn metric.
        Returns:
            Induced metric (and index of dependent coordinate if None).
        """
        grad_def = self.grad_defining(zs, params, patch)

        return_dep = False
        if dependent is None:
            dependent = jnp.argmax(jnp.abs(grad_def), axis=-1)
            return_dep = True

        if zs_c is None:
            ind_met = induced_metric(metric, grad_def, dependent)
        else:
            grad_def_c = self.grad_defining(zs_c, jnp.conj(params), patch)
            ind_met = induced_metric(metric, grad_def, dependent, grad_def_c)

        return (ind_met, dependent) if return_dep else ind_met

    def induced_fs(self, zs, params, patch, zs_c=None, dependent=None):
        """Induced FS metric.
        Returns:
            The induced Fubini-Study metric (and dependent coordinate if None).
        """
        metric = fs_metric(zs, zs_c)
        return self.induced_metric(metric, zs, params, patch, zs_c, dependent)

    @auto_vmap(zs=1, patch=0, params=1, dependent=0)
    def sample_intersect_weights(self, zs, params, patch, dependent=None, vol=1, separate=False):
        """Compute weights dVolCy / dA.

        dA here is the measure with respect to the pullback of the ambient
        FS metric while dVolCy comes from the holomorphic volume form.
        Args:
            zs: Affine coordinates.
            patch: Patch affine coordinates are defined in.
            variety: Variety on which points lie.
            par: Parameters (complex moduli) of variety.
            dependent: Which affine coordinate index to treat as dependent.
            separate: If true return tuple (dVolCy, dA).
        """
        # weights are independent of choice of coordinates.
        grad_def = self.grad_defining(zs, params, patch)
        # index not in absolute terms
        if dependent is None:
            dependent = jnp.argmax(jnp.abs(grad_def))

        grad_dep, grad_indep = util.pop(grad_def, dependent)
        # note: not called with dedicated zs_c as we won't differentiate
        g_fs = fs_metric(zs)
        jac = util.insert_col(
            jnp.eye(self.dim_variety, dtype=complex),
            -grad_indep / grad_dep, dependent)
        g_ind = jac @ g_fs @ jnp.transpose(jac).conj()

        det = jnp.linalg.det(g_ind).real
        dvol = 1 / jnp.abs(grad_dep) ** 2
        if vol is not None:
            dim = self.dim_variety
            det *= factorial(dim) / np.pi ** dim
            dvol *= vol
        if separate:
            return dvol, det
        return 1 / (det * jnp.abs(grad_dep) ** 2)

    def sample_intersect(self, key: chex.PRNGKey, params, count: int, affine=False, weights=False):
        """Sample points on the variety using the intersection method.

        Args:
            key: Random key.
            count: Number of samples to generate.
            params: Numeric parameter values of the defining equation.
            affine: Whether to convert output to affine coordinates.
            weights: Whether to compute the Monte Carlo weights for the
                generated samples.

        Returns:
            Either the homogeneous coordinates or a tuple of affine coordinates
            and patch indices.
        """
        ps = sympy.symarray('p', self.dim_complex)
        qs = sympy.symarray('q', self.dim_complex)
        t = np.array(sympy.Symbol('t'))
        z = ps + t * qs
        poly = self.defining_poly(z, np.array(self.parameters))
        poly = Poly.from_sympy(poly, [t, ps, qs], self.parameters)

        sample_count = np.ceil(count / self.dim_complex).astype(int)
        ps, qs = complex_sphere(key, (2, sample_count, self.dim_complex))
        coefficients = univariate_coefficients(poly, t)
        zs_hom = _solve_intersect(ps, qs, params, coefficients)
        zs_hom = zs_hom.reshape(-1, self.dim_complex)[:count]

        if affine:
            zs, patch = hom_to_affine(zs_hom)
            if weights:
                zs_weight = self.sample_intersect_weights(zs, params, patch)
                return (zs, patch), zs_weight
            else:
                return zs, patch
        elif weights:
            zs, patch = hom_to_affine(zs_hom)
            zs_weight = self.sample_intersect_weights(zs, params, patch)
            return zs_hom, zs_weight
        else:
            return zs_hom

    def compute_vol(self, key, params, batches=200, batch_size=2000, var=False):
        """Compute CY (top-form) volume of manifold.
        If var is True, also return variance.
        """
        def sample(key, count):
            return self.sample_intersect(key, params, count, False, True)

        return _int(key, batches, batch_size, lambda zs: 1, sample, var)

    @auto_vmap(zs=1, params=1, patch=0, dependent=0)
    def solve_defining(self, zs, params, dependent, patch=None):
        """Solve defining equation given n-1 points.

        Given n-1 coordinates, solve for the nth coordinate.

        The parameters shuffle_key and patch_size should only be used when
        the variety is symmetric under coordinate permutations.

        Args:
            zs: Local coordinates in given patch where dependent is omitted.
            patch: Affine patch within which local coordinates are given.
            dependent: Dependent coordinate; range 0, ..., dim_proj.
            params: Parameters of variety.
        """
        # solve defining equation given all but one coordinate
        if patch is None:
            defining_poly = self.defining_poly
        else:
            defining_poly = self.defining_poly.affine_poly(patch)

        dep_hom = index_affine_to_hom(patch, dependent)
        dep = self.defining_poly.all_symbols(True, False)[0][dep_hom]
        # get coefficient polynomials
        coeffs = univariate_coefficients(defining_poly, dep)
        # evaluate coefficients
        coeffs = jnp.array([
            coeff(zs, params) if isinstance(coeff, Poly) else coeff
            for coeff in coeffs])
        # solve for missing coordinate value
        roots = jnp.roots(coeffs, strip_zeros=False)

        # insert roots into solution
        pt = jnp.roll(zs, -dependent)
        pt = jax.vmap(jnp.append, in_axes=(None, 0))(pt, roots)
        pts = jnp.roll(pt, dependent + 1, 1)

        return pts

    def __repr__(self):
        return f'{type(self).__name__}(dim={self.dim}) ' \
               f'in CP^{self.dim_projective}: {repr(self.defining_poly)}'

    def _repr_latex_(self):
        name = '{' + type(self).__name__ + '}'
        return rf'$\operatorname{name}(dim={self.dim}) ' \
               rf'\subset \mathbb{{CP}}^{self.dim_projective}: ' \
               + self.defining_poly._repr_latex_()[1:]


class Fermat(VarietySingle):
    def __init__(self, dim_variety=3):
        poly = FermatPoly(dim_variety + 2)
        super(Fermat, self).__init__(poly)

    @classmethod
    def from_sympy(cls, _: Union[str, sympy.Expr, sympy.Poly],
                   var_name: str = 'z', poly_degree: int = 5,
                   params: Sequence[Union[sympy.Symbol, str]] = None):
        raise NotImplementedError


class Dwork(VarietySingle):
    def __init__(self, dim_variety=3, factor=True):
        poly = DworkPoly(dim_variety + 2, factor)
        super(Dwork, self).__init__(poly)

    @classmethod
    def from_sympy(cls, _: Union[str, sympy.Expr, sympy.Poly],
                   var_name: str = 'z', poly_degree: int = 5,
                   params: Sequence[Union[sympy.Symbol, str]] = None):
        raise NotImplementedError
