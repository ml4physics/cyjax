import jax
import jax.numpy as jnp

from .util import *
from cyjax import *
from itertools import combinations


class RandomTests(TestCase):
    def test_uniform_components(self):
        key = jax.random.PRNGKey(42)
        shape = (10, 7)
        zs = random.uniform_components(key, shape)
        self.assertEqual(zs.shape, shape)
        self.assertAllTrue(jnp.abs(zs.real) < 1)
        self.assertAllTrue(jnp.abs(zs.imag) < 1)

    def test_uniform_angle_unit(self):
        key = jax.random.PRNGKey(42)
        shape = (10, 7)
        zs = random.uniform_angle_unit(key, shape)
        self.assertEqual(zs.shape, shape)
        self.assertAllClose(jnp.abs(zs), 1)

    def test_uniform_angle(self):
        key = jax.random.PRNGKey(42)
        shape = (10, 7)
        max_abs = 0.5
        zs = random.uniform_angle(key, shape, 0, max_abs)
        self.assertEqual(zs.shape, shape)
        self.assertAllTrue(jnp.abs(zs) < max_abs)

    def test_real_sphere(self):
        key = jax.random.PRNGKey(42)
        shape = (10, 6, 7)
        zs = random.real_sphere(key, shape)
        self.assertEqual(zs.shape, shape)
        self.assertAllClose(jnp.linalg.norm(zs, axis=-1), 1)

    def test_complex_sphere(self):
        key = jax.random.PRNGKey(42)
        shape = (10, 6, 7)
        zs = random.complex_sphere(key, shape)
        self.assertEqual(zs.shape, shape)
        self.assertAllClose(jnp.linalg.norm(zs, axis=-1), 1)

    def test_uniform_projective(self):
        key = jax.random.PRNGKey(42)
        count = 200
        dim = 7
        zs, patches = random.uniform_projective(key, count, dim)
        self.assertEqual(zs.shape, (count, dim))
        self.assertEqual(patches.shape, (count,))
        chex.assert_type(patches, int)
        self.assertAllTrue(jnp.abs(zs) < 1)

    def test_overlap(self):
        key = jax.random.PRNGKey(42)
        patch_size = 1.1
        for i, j in combinations(range(8), 2):
            zs = random.projective_overlap(key, 100, 7, i, j, patch_size)
            self.assertAllTrue(jnp.abs(zs) < patch_size)
            self.assertInOverlap(zs, i, j, patch_size)

            zs = random.projective_overlap(key, 100, 7, j, i, 1.1)
            self.assertAllTrue(jnp.abs(zs) < patch_size)
            self.assertInOverlap(zs, j, i, patch_size)
