import jax.random
from jax.config import config
config.update("jax_enable_x64", True)

import sympy
from .util import *
from cyjax import *


# an example polynomial
example_poly_expr = \
    'a * z0**5 + b * z1**5 + c * z2**5 + d * z3**5 + e * z4**5 ' \
    '- 5 * psi * z0 * z1 * z2 * z3 * z4'

# intentionally flip c, b to check custom order works
param_names = ['a', 'c', 'b', 'd', 'e', 'psi']

example_poly_powers = (  # variables; here, only one
    [  # monomials / terms to sum
        [5, 0, 0, 0, 0],  # powers for each variable index
        [0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 5],
        [1, 1, 1, 1, 1],
    ],
)

example_poly_coeff = [1, 1, 1, 1, 1, -5]
example_poly_par_coeff = [['a'], ['b'], ['c'], ['d'], ['e'], ['psi']]


def example_poly(z, params):
    a, c, b, d, e, psi = params

    return (a * z[0]**5 + b * z[1]**5 + c * z[2]**5 + d * z[3]**5 +
            e * z[4]**5 - 5 * psi * z[0] * z[1] * z[2] * z[3] * z[4])


class Bases:

    class PolyTest(TestCase):
        poly: Poly

        def test_expr(self: TestCase):
            # compare generated sympy expression with manual construction
            params = [sympy.Symbol(s) for s in param_names]
            domain = sympy.CC[params]
            sympy_poly = sympy.Poly(example_poly_expr, domain=domain)
            self.assertEqual(sympy_poly, self.poly.sympy_poly)

        def test_single(self: TestCase):
            # check evaluation for single value of z and params
            num_var = len(self.poly.variable_indices[0])
            z, = random.uniform_projective(self.next_key(), 1, num_var, False)
            params = random.uniform_angle(self.next_key(), (len(param_names),))
            v1 = self.poly(z, params)
            v2 = self.poly(z, params=params)
            v3 = self.poly(params=params, z=z)
            truth = example_poly(z, params)
            self.assertAllClose(truth, v1)
            self.assertAllClose(truth, v2)
            self.assertAllClose(truth, v3)

        def test_symbolic(self: TestCase):
            # check that symbolic input works
            num_var = len(self.poly.variable_indices[0])
            z = sympy.symarray('z', num_var)
            params = [sympy.Symbol(s) for s in param_names]
            params[0] = 0
            expr = example_poly(z, params)
            self.assertTrue(expr.equals(self.poly(z, params)))

        def test_multi(self: TestCase):
            # check evaluation for batch values of z and params
            num_var = len(self.poly.variable_indices[0])
            zs = random.uniform_projective(self.next_key(), 10, num_var, False)
            params = random.uniform_angle(self.next_key(), (10, len(param_names)))

            # get function that takes batch inputs
            eval_fn = self.poly.transform_eval()

            truth = example_poly(zs.T, params.T)
            v1 = eval_fn(zs, params)
            v2 = eval_fn(params=params, z=zs)
            self.assertAllClose(truth, v1)
            self.assertAllClose(truth, v2)

            # single params, multi zs
            truth = example_poly(zs.T, params[0])
            v1 = eval_fn(zs, params[0])
            self.assertAllClose(truth, v1)


class PolyTestSympy(Bases.PolyTest):
    def setUp(self):
        super().setUp()
        self.poly = Poly.from_sympy(example_poly_expr, ['z'], param_names)


class PolyTestStruct(Bases.PolyTest):
    def setUp(self):
        super().setUp()
        self.poly = Poly.from_coeffs_and_powers(
            example_poly_coeff, example_poly_powers, example_poly_par_coeff,
            parameters=param_names, variable_names=['z'])


class HomPolyTest(Bases.PolyTest):
    def setUp(self):
        super().setUp()
        self.poly = polynomial.HomPoly.from_sympy(
            example_poly_expr, ['z'], param_names)

    def test_affine(self):
        # test evaluation in terms of affine coordinates if patch is given
        eval_fn = self.poly.transform_eval()

        zs = random.uniform_projective(self.next_key(), 100, 5, False)
        params = random.uniform_angle(self.next_key(), (len(param_names),))

        for patch in range(5):
            zs_hom = zs.at[:, patch].set(1)
            zs_aff = hom_to_affine(zs_hom, patch)
            v0 = example_poly(zs_hom.T, params)
            v1 = eval_fn(zs_hom, params)
            v2 = eval_fn(zs_aff, params, patch=patch)
            v3 = eval_fn(zs_aff, params, patch=(patch,))
            s1 = self.poly(zs_hom[0], params)
            s2 = self.poly(zs_aff[0], params, patch=patch)
            self.assertAllClose(v0, v1)
            self.assertAllClose(v0, v2)
            self.assertAllClose(v0, v3)
            self.assertAllClose(v0[0], s1)
            self.assertAllClose(v0[0], s2)

        # multiple patches
        patch = jax.random.randint(self.next_key(), (100,), 0, 5)
        zs_hom = zs.at[np.arange(100), patch].set(1)
        zs_aff = hom_to_affine(zs_hom, patch)
        self.assertAllClose(example_poly(zs_hom.T, params),
                            eval_fn(zs_aff, params, patch=patch))
        self.assertAllClose(example_poly(zs_hom.T, params),
                            eval_fn(zs_aff, params, patch=(patch,)))
