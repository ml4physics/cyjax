import jax.random
import jax.numpy as jnp
import cyjax as jcm

from cyjax import *
from .util import *
from itertools import combinations


class ProjectiveTests(TestCase):

    def test_change_chart(self):
        key = jax.random.PRNGKey(32)

        for i, j in combinations(range(8), 2):
            pts = jcm.random.projective_overlap(key, 100, 7, i, j, 1.1)
            self.assertAllTrue(jnp.abs(change_chart(pts, i, j)) < 1.1)
            pts = jcm.random.projective_overlap(key, 100, 7, j, i, 1.1)
            self.assertAllTrue(jnp.abs(change_chart(pts, j, i)) < 1.1)

    def test_index_from_global(self):
        self.assertEqual(index_hom_to_affine(0, 1), 0)
        self.assertEqual(index_hom_to_affine(2, 1), 1)

    def test_index_to_global(self):
        self.assertEqual(index_affine_to_hom(0, 0), 1)
        self.assertEqual(index_affine_to_hom(1, 0), 0)
        self.assertEqual(index_affine_to_hom(0, 1), 2)

    def test_hom_to_affine(self):
        hom = jnp.array([[1.0, 0.2, 0.3]])
        zs, p = hom_to_affine(hom)
        self.assertTrue(p.reshape(()) == 0)
        self.assertAllEqual(hom[:, 1:], zs)

        hom = jnp.array([[1.0, 3.2, 0.3]])
        zs, p = hom_to_affine(hom)
        self.assertTrue(p.reshape(()) == 1)
        self.assertAllEqual(jnp.array([1, .3])/3.2, zs)

        zs = hom_to_affine(hom, 0)
        self.assertAllEqual(hom[:, 1:], zs)

    def test_affine_to_hom(self):
        hom = jcm.random.uniform_angle(jax.random.PRNGKey(42), (100, 7))
        hom_back = affine_to_hom(*hom_to_affine(hom))
        hom_normed = hom / jnp.max(jnp.abs(hom), 1, keepdims=True)
        self.assertAllClose(jnp.abs(hom_back / hom_normed), 1)
