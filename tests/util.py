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

import chex
import jax.random
import numpy as np
import jax.numpy as jnp
from time import time_ns


class TestCase(chex.TestCase):

    def setUp(self):
        seed = time_ns()
        self.init_random(seed)

    def init_random(self, seed: int):
        self._key = jax.random.PRNGKey(seed)

    def next_key(self, num=1):
        self._key, *key = jax.random.split(self._key, num=num+1)
        return key[0] if num == 1 else key

    def assertAllEqual(self, x, y):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.alltrue(x == y))

    def assertAllClose(self, x, y, rtol=1e-5, atol=1e-8):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(jnp.allclose(x, y, rtol=rtol, atol=atol))

    def assertAllTrue(self, t):
        self.assertTrue(jnp.alltrue(t))

    def assertInOverlap(self, zs, this_chart, other_chart, patch_size):
        other = other_chart - (this_chart < other_chart)
        norms = np.abs(zs)
        others_small = norms <= patch_size * norms[:, other].reshape(-1, 1)
        large_enough = norms[:, other] >= 1 / patch_size
        self.assertAllTrue(others_small)
        self.assertAllTrue(large_enough)
