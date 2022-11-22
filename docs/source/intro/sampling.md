# Sampling points
To numerically evaluate integrals, e.g. to compute the volume, we use a Monte Carlo approximation.
We thus need to generate samples that lie on the manifold and have a known distribution.
One straight-forward way of generating points on the $d$ dimensional manifold is to sample $d-1$ random complex numbers and solve the defining equation for the last coordinate value.
However, we do not apriori know the distribution of these points.
Instead, we can sample points as intersections with a line in ambient projective space.
The distribution of these points is known.

## Complex and projective coordinates
Some care needs to be taken when sampling complex coordinates.
If real and imaginary part are sampled uniformly, points will lie on a square in the complex plane.
Uniformly sampling the radius and the complex angle will ensure points lie on the unit disk, however the distribution does not have uniform density.
Instead, one should sample uniformly from the disk in $\mathbb{R}^2$ and interpret the coordinates as real and imaginary parts.

In order to generate samples from $\mathbb{P}^n$ we sample $2n+2$ real numbers on the real sphere $S^{2n+1}$.
By pairing up real numbers into pairs of two into a complex value we obtain homogeneous coordinates on the projective space.
This construction corresponds to $\mathbb{P}^n \cong S^{2n+1} / U(1)$.
Generating points on the real sphere $S^{2n+1}$ can be done efficiently by independently sampling $2n+2$ real numbers from the normal distribution and dividing by their vector norm.
This works due to the factorization of the normal distribution $\exp(- \sum_i z_i^2)=\prod_i \exp(-z_i^2)$, where the former is clearly symmetric under the $SO(2n+2)$ symmetry which ensures a uniform distribution.

```{eval-rst}
Functions implementing the above are grouped in the :py:mod:`cyjax.random` submodule:

.. currentmodule:: cyjax

.. autosummary::

    random.uniform_components
    random.uniform_angle
    random.uniform_angle_unit
    random.real_sphere
    random.complex_sphere
    random.uniform_projective
    random.projective_overlap
```

## Points on varieties

To numerically estimate integrals, we make use of the Monte Carlo approximation
$$
\int_X f d\mathrm{vol} = \int_X f \frac{d\mathrm{vol}}{dA} dA \approx \frac{1}{N} \sum_{a=1}^{N} f(x_a) w(x_a) \,.
$$
Here, $x_a \sim a$ are drawn using some (pseud-) probabilistic procedure with known density measure $dA$.
The weights $w(x_a)$ in the final step are required to correct for the difference in measure.

The sampling method we use here is described in [(Douglas et al. 2008)](https://doi.org/10.1063/1.2888403).
After uniformly sampling two points $p, q \in \mathbb{P}^{d+1}$, we can define a line $p + t q$ with $t\in\mathbb{C}$.
Samples on the variety are then given by the intersection of this line with the variety, i.e. by solutions for $t$ such that $Q(p+tq)=0$.
The density of samples generated in this way is known and given via the Fubini-Study metric.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    VarietySingle.sample_intersect  
    VarietySingle.sample_intersect_weights
    VarietySingle.compute_vol
```
