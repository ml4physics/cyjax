import jax
import jax.numpy as jnp
from .util import *
from cyjax import util


class UtilTests(TestCase):

    def test_pop(self):
        l = [0, 1, 2, 3]
        for i in l:
            j, x = util.pop(jnp.array(l), i)
            self.assertEqual(j, i)
            self.assertTrue(
                jnp.allclose(x, jnp.array([k for k in l if k != i])))

        x = jnp.array([[0, 1], [2, 3], [4, 5]])
        c0, y0 = util.pop(x, 0)
        c1, y1 = util.pop(x, 1)
        c2, y2 = util.pop(x, 2)
        self.assertAllEqual(c0, x[0])
        self.assertAllEqual(c1, x[1])
        self.assertAllEqual(c2, x[2])
        self.assertAllEqual(y0, x[1:])
        self.assertAllEqual(y1, jnp.stack([x[0], x[2]]))
        self.assertAllEqual(y2, x[:-1])

    def test_insert_1d(self):
        x = jnp.array([0, 1, 2, 3.0])
        x0 = jnp.array([-1, 0, 1, 2, 3.0])
        x1 = jnp.array([0, -1, 1, 2, 3.0])
        x2 = jnp.array([0, 1, -1, 2, 3.0])
        x3 = jnp.array([0, 1, 2, -1, 3.0])
        x4 = jnp.array([0, 1, 2, 3.0, -1])
        for i, y in enumerate([x0, x1, x2, x3, x4]):
            self.assertAllEqual(util.insert_1d(x, -1, i), y)
        for i, y in [(-1, x4), (-2, x3), (-3, x2), (-4, x1),(-5, x0)]:
            self.assertAllEqual(util.insert_1d(x, -1, i), y)

    def test_insert_col(self):
        m = jnp.array([[0, 1], [2, 3]])
        c = jnp.array([-1, -2])
        self.assertAllEqual(util.insert_col(m, c, 0), jnp.array([[-1, 0, 1], [-2, 2, 3]]))
        self.assertAllEqual(util.insert_col(m, c, 1), jnp.array([[0, -1, 1], [2, -2, 3]]))
        self.assertAllEqual(util.insert_col(m, c, 2), jnp.array([[0, 1, -1], [2, 3, -2]]))
        self.assertAllEqual(util.insert_col(m, c, -1), jnp.array([[0, 1, -1], [2, 3, -2]]))
        self.assertAllEqual(util.insert_col(m, c, -2), jnp.array([[0, -1, 1], [2, -2, 3]]))
        self.assertAllEqual(util.insert_col(m, c, -3), jnp.array([[-1, 0, 1], [-2, 2, 3]]))

    def test_permute_axis(self):
        key = jax.random.PRNGKey(10)
        for _ in range(3):
            k1, k2, key = jax.random.split(key, 3)
            arr = jax.random.choice(k1, jnp.arange(100), (5, 7, 10))
            arr_p = util.shuffle_axis(k2, arr, 1)
            for i, (a, a_p) in enumerate(zip(arr, arr_p)):
                # make sure sub-arrays are preserved
                a_set = set(tuple(int(i) for i in row) for row in a)
                a_p_set = set(tuple(int(i) for i in row) for row in a)
                self.assertEqual(a_set, a_p_set)

    def test_binomial(self):
        self.assertEqual(util.binomial(10, 1), 10)
        self.assertEqual(util.binomial(10, 10), 1)
        self.assertEqual(util.binomial(10, 0), 1)
        self.assertEqual(util.binomial(4, 2), 4 * 3 // 2)
