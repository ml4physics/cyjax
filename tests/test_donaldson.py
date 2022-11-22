import jax
from functools import partial
from .util import *
import cyjax


class TestGeometry(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.degree = 6
        cls.variety = cyjax.Dwork()
        cls.params = jnp.array([0.5 + 0.02j])
        cls.monomials = cyjax.donaldson.MonomialBasisReduced(
            cls.variety, cls.degree, cls.params)

        key = jax.random.PRNGKey(10)
        zs_hom, = jax.jit(
            partial(
                cls.variety.sample_intersect,
                count=1, params=cls.params, affine=False),
            backend='cpu')(key)

        zs, patch = cyjax.hom_to_affine(zs_hom)
        cls.z = zs
        cls.z_c = jnp.conj(zs)
        cls.patch = patch
        h = jnp.eye(cls.monomials.size) + 0j

        cls.go = cyjax.donaldson.GeometricObjects(
            h, zs_hom, cls.params, cls.variety, cls.monomials)

        @jax.jit
        def kahler(z, z_c):
            s = cls.monomials(z, patch)
            s_c = cls.monomials(z_c, patch)
            psi = jnp.einsum('i,ij,j', s, h, s_c)
            return jnp.log(psi) / (jnp.pi * cls.degree)

        cls.kahler = staticmethod(kahler)
        metric = cyjax.complex_hessian(kahler)
        cls.metric = staticmethod(metric)

        def jac_front(fun, index):
            @jax.jit
            def wrapped(z, z_c):
                # by default jac adds grad dimension to the back but
                # here want them in the front
                out = jax.jacfwd(fun, index, holomorphic=True)(z, z_c)
                return jnp.moveaxis(out, -1, 0)
            return wrapped

        metric_h = jac_front(metric, 0)
        cls.metric_h = staticmethod(metric_h)
        cls.metric_ah = staticmethod(jac_front(metric_h, 1))

        @jax.jit
        def metric_loc(z, z_c):
            g_proj = cls.metric(z, z_c)
            g, _ = cls.variety.induced_metric(
                g_proj, z, cls.params, cls.patch, zs_c=z_c)
            return g

        @jax.jit
        def logdet(z, z_c):
            g = metric_loc(z, z_c)
            return jnp.log(jnp.linalg.det(g))

        cls.logdet = staticmethod(logdet)
        cls.metric_loc = staticmethod(metric_loc)
        cls.metric_loc_h = staticmethod(jac_front(metric_loc, 0))

    def test_dependent(self):
        dep = self.go.dependent
        chex.assert_type(dep, int)
        chex.assert_shape(dep, ())

    def test_patch(self):
        self.assertEqual(self.patch, self.go.patch)

    def test_zs(self):
        self.assertAllEqual(self.z, self.go.zs)

    def test_s(self):
        chex.assert_shape(self.go.s, (self.monomials.size,))

    def test_s_1(self):
        shape = (self.monomials.size, self.variety.dim_projective)
        chex.assert_shape(self.go.s_1, shape)

    def test_s_2(self):
        shape = (self.monomials.size,
                 self.variety.dim_projective,
                 self.variety.dim_projective)
        chex.assert_shape(self.go.s_2, shape)

    def test_s_c(self):
        self.assertAllEqual(self.go.s_c, jnp.conj(self.go.s))

    def test_psi(self):
        chex.assert_shape(self.go.psi, ())

    def test_kahler(self):
        self.assertAllClose(self.go.kahler, self.kahler(self.z, self.z_c))

    def test_psi_h(self):
        shape = (self.variety.dim_projective,)
        chex.assert_shape(self.go.psi_h, shape)

    def test_psi_ha(self):
        shape = (self.variety.dim_projective,) * 2
        chex.assert_shape(self.go.psi_ha, shape)

    def test_psi_hh(self):
        shape = (self.variety.dim_projective,) * 2
        chex.assert_shape(self.go.psi_aa, shape)

    def test_psi_aa(self):
        shape = (self.variety.dim_projective,) * 2
        chex.assert_shape(self.go.psi_aa, shape)

    def test_psi_hha(self):
        shape = (self.variety.dim_projective,) * 3
        chex.assert_shape(self.go.psi_hha, shape)

    def test_psi_hhaa(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.psi_hhaa, shape)

    def test_g_proj(self):
        g = self.metric(self.z, self.z_c)
        self.assertAllClose(g, self.go.g_proj)

    def test_psi_h_psi_ha(self):
        shape = (self.variety.dim_projective,) * 3
        chex.assert_shape(self.go.psi_h_psi_ha, shape)

    def test_two_hha(self):
        shape = (self.variety.dim_projective,) * 3
        chex.assert_shape(self.go.two_hha, shape)

    def test_three_hha(self):
        shape = (self.variety.dim_projective,) * 3
        chex.assert_shape(self.go.three_hha, shape)

    def test_g_h(self):
        g_h = self.metric_h(self.z, self.z_c)
        self.assertAllClose(g_h, self.go.g_proj_h)

    def test_psi_hha_psi_a(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.psi_hha_psi_a, shape)

    def test_psi_ha_psi_ha(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.psi_ha_psi_ha, shape)

    def test_two_haha(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.two_haha, shape)

    def test_three_haha(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.three_haha, shape)

    def test_four_haha(self):
        shape = (self.variety.dim_projective,) * 4
        chex.assert_shape(self.go.four_haha, shape)

    def test_g_ha(self):
        g_ah = self.metric_ah(self.z, self.z_c)
        g_ha = jnp.swapaxes(g_ah, 1, 0)
        self.assertAllClose(g_ha, self.go.g_proj_ha)

    def test_grad_def(self):
        shape = (self.variety.dim_projective,)
        chex.assert_shape(self.go.grad_def, shape)

    def test_jac(self):
        shape = (self.variety.dim, self.variety.dim_projective)
        chex.assert_shape(self.go.jac, shape)

    def test_jac_h(self):
        shape = (self.variety.dim,
                 self.variety.dim_projective,
                 self.variety.dim_projective)
        chex.assert_shape(self.go.jac_h, shape)

    def test_g_loc(self):
        g = self.metric_loc(self.z, self.z_c)
        self.assertAllClose(self.go.g_loc, g)

    def test_g_loc_h(self):
        g_loc_h = self.metric_loc_h(self.z, self.z_c)
        self.assertAllClose(g_loc_h, self.go.g_loc_h)

    def test_g_loc_ha(self):
        shape = (self.variety.dim_projective,) * 2 + (self.variety.dim,) * 2
        chex.assert_shape(self.go.g_loc_ha, shape)

    def test_g_loc_inv(self):
        self.assertAllClose(self.go.g_loc_inv @ self.go.g_loc,
                            np.eye(self.variety.dim))

    def test_ricci_proj(self):
        ricci = -cyjax.complex_hessian(self.logdet)(self.z, self.z_c)
        self.assertAllClose(ricci, self.go.ricci_proj)

    def test_ricci_loc(self):
        shape = (self.variety.dim,) * 2
        chex.assert_shape(self.go.ricci_loc, shape)

    def test_ricci_scalar(self):
        chex.assert_shape(self.go.ricci_scalar, ())

    def test_eta(self):
        self.assertAllClose(self.go.eta.imag, 0)
        chex.assert_shape(self.go.eta, ())


class DonaldsonTests(TestCase):

    def test_donaldson(self):
        niter = 10
        batch_size = 1000
        degree = 2
        psi = jnp.array([10 + 3j])  # fix parameter
        dwork = cyjax.Dwork(3)

        volcy = jax.jit(
            partial(dwork.compute_vol, batch_size=500), backend='cpu'
        )(self.next_key(), psi)

        mon_basis = cyjax.donaldson.MonomialBasisReduced(dwork, degree, psi)
        alg_metric = cyjax.donaldson.AlgebraicMetric(dwork, mon_basis)

        Nk = mon_basis.size
        # an estimate for how many MC sample points to use in integral
        Np = (10 * Nk**2 + 50000) // 5
        batches = Np // batch_size + 1

        # init
        h = jnp.eye(Nk, dtype=complex)

        # JIT compile procedure with hyperparameters fixed
        step = jax.jit(
            partial(
                alg_metric.donaldson_step,
                params=psi, vol_cy=volcy, batches=batches, batch_size=batch_size),
            backend='cpu')

        h_iter = h
        for i in range(niter):
            h_iter = (h_iter + h_iter.conj().T) / 2  # assure h is Hermitian
            h_iter = step(self.next_key(), h_iter)
            h_iter = h_iter / jnp.max(jnp.abs(h_iter))

        eta_acc = jax.jit(partial(alg_metric.sigma_accuracy, count=1000), backend='cpu')
        sig = eta_acc(self.next_key(), psi, h_iter)
        self.assertTrue(0 < sig < 1)
