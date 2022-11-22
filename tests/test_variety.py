from jax.config import config
config.update("jax_enable_x64", True)

from .util import *
from cyjax import *
from functools import partial

class SamplingTests(TestCase):

    def test_intersect_fermat(self):
        # sample points and check defining equation is solved
        par = None

        for dim in [1, 3]:
            fermat = Fermat(dim)

            sample = partial(fermat.sample_intersect, count=30, params=par)
            sample = jax.jit(sample, static_argnames=('affine',), backend='cpu')
            zs = sample(self.next_key())
            def_eqn = fermat.defining(zs, par)
            self.assertAllClose(def_eqn, 0, atol=1e-5)

            zs, patch = sample(self.next_key(), affine=True)
            def_eqn = fermat.defining(zs, par, patch)
            self.assertAllClose(def_eqn, 0, atol=1e-5)

    def test_intersect_dwork(self):
        # sample points and check defining equation is solved
        par = jnp.array([1j - 3])

        for dim in [1, 3]:
            dwork = Dwork(dim)

            sample = partial(dwork.sample_intersect, count=30, params=par)
            sample = jax.jit(sample, static_argnames=('affine',), backend='cpu')
            zs = sample(self.next_key())
            def_eqn = dwork.defining(zs, par)
            self.assertAllClose(def_eqn, 0, atol=1e-5)

            zs, patch = sample(self.next_key(), affine=True)
            def_eqn = dwork.defining(zs, par, patch)
            self.assertAllClose(def_eqn, 0, atol=1e-5)

    def test_intersect_sympy(self):
        # sample points and check defining equation is solved
        par = jnp.array([1.2, 1.4, 1, 0.3, 3, 1.1])

        var = VarietySingle.from_sympy(
            'a * z_0**5 + b * z_1**5 + c * z_2**5 + d * z_3**5 + e * z_4**5 - 5 * psi * z_0 * z_1 * z_2 * z_3 * z_4',
            'z', 5)

        sample = partial(var.sample_intersect, count=30, params=par)
        sample = jax.jit(sample, static_argnames=('affine',), backend='cpu')

        zs = sample(self.next_key())
        def_eqn = var.defining(zs, par)
        self.assertAllClose(def_eqn, 0, atol=1e-5)

        zs, patch = sample(self.next_key(), affine=True)
        def_eqn = var.defining(zs, par, patch)
        self.assertAllClose(def_eqn, 0, atol=1e-5)

    def test_batch_sampler(self):
        def psi_sampler(key, batch):
            return random.uniform_angle(key, (batch, 1))

        var = Dwork(3)

        batch_size = 100
        batch_size_params = 5

        batch_sampler = ml.BatchSampler(
            variety=var,
            params_sampler=psi_sampler,
            batch_size_params=batch_size_params,
            batch_size=batch_size,
            buffer_size=2,
            seed=self.next_key())

        psi, zs, patch, weights = next(batch_sampler)
        self.assertEqual(psi.shape, (batch_size_params, 1))
        self.assertEqual(zs.shape, (batch_size_params, batch_size, var.dim_projective))
        self.assertEqual(patch.shape, (batch_size_params, batch_size))
        self.assertEqual(weights.shape, (batch_size_params, batch_size))

        psi = jnp.repeat(psi.reshape(batch_size_params, 1, 1), 100, 1)
        def_eqn = jax.jit(var.defining)(zs, psi, patch)
        self.assertAllClose(jnp.abs(def_eqn), 0, atol=1e-5)
