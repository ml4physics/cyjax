# Complex projective varieties
We consider here varieties $X \subset \mathbb{P}^{d+1}$ which are defined as the zero-locus of a single homogeneous polynomial:
$$ 
Q(z) = \sum_{\alpha} \psi_\alpha z^\alpha \Bigg|_{z \in X} = 0 \,.
$$
Recall that the homogeneous coordinates $z=[z_0: \ldots: z_{d+1}]$ of projective space $\mathbb{P}^{d+1}$ are just the complex coordinates of $\mathbb{C}^{d+2}$ with the identification $z \sim \lambda z$ for all $\lambda \in \mathbb{C} \setminus \{0\}$.
The multi-index $\alpha$ ranges over all natural numbers such that $\sum_{i=0}^{d+1} \alpha_i = d+2$, corresponding to all monomials of degree $d+2:$
$$
z^\alpha = \prod_{i=0}^{d+1} z_i^{\alpha_i} \,.
$$
Choosing particular values of coefficients $\psi_\alpha \in \mathbb{C}$ corresponds to fixing the complex moduli.

The dimension of the variety $X$ in the above example is $d$. 
Of particular interest in physics is the case $d=3$, where $X$ is called a quintic threefold (referring to the degree of the defining equation and manifold's dimension, respectively).
We may restrict which coefficients $\psi$ we allow to be nonzero.
A particular example is the so-called Dwork family of quintics:
$$
\sum_{i=0}^{4} z_i^{5} + \psi \prod_{i=0}^{4} z_i \,.
$$

One can show that the first Chern class vanishes for the above manifolds.
We thus know a Ricci flat-metric exists and want to find a numerical approximation to it.
The problem of fixing Kähler moduli is particularly simple for the quintic.
Since there is a single Kähler modulus, it can be chosen by rescaling the volume of any final metric.

## Choice of coordinates
There is no unique choice for going from the $d+2$ coordinates in the ambient complex space to "true" coordinates on the $d$-dimensional variety $X$.
One option would be to keep the full redundancy and stick with the homogeneous coordinates.
However, some geometric quantities have no numeric globally defined representation, and thus choices about coordinate patches have to be made.

### Projective coordinates
To get rid of the scaling ambiguity in homogeneous coordinates, we pick one index (with non-zero entry) and set its value to $1$ by rescaling:
$$
[z_0: \ldots: z_p:  \ldots: z_{d+1}] 
&\sim [z_0/z_p: \ldots: 1:  \ldots: z_{d+1}/z_p] \\
&\equiv (z_0/z_p, \ldots z_{d+1}/z_p)
= (z^{(p)}_0, \ldots, z^{(p)}_{d}) = z^{(p)} \,.
$$
If index $p$ of the $d+2$ homogeneous coordinates is scaled to one, we denote the remaining $d+1$ *affine coordinates* by $z^{(p)}$. 
Computationally, the affine coordinates are represented by an array with $d+1$ entries together with an integer specifying the patch they are in (i.e. which homogeneous index was scaled  to $1$).

Going from homogeneous coordinates to patch $p$, we divide by the value of $z_p$.
Numerically, it is advantageous to avoid very large values.
For the numerically "optimal" patch we thus typically choose $p$ such that $|z_p|$ is maximal.

```{warning}
Note that python (and thus the code here) uses [zero-based indexing](https://en.wikipedia.org/wiki/Zero-based_numbering) while mathematical notation usually uses one-based indexing.
For consistency, the indexing in the notes here also starts at $0$.
For example, ``z_aff[1]`` which indexes into the affine coordinates of patch 0, $z^{(0)}=(z_1/z_0, z_2/z_0, \ldots)$, would give $z_2/z_0$.
But homogeneous coordinates are equivalent up to rescaling so we can just forget about $z_0$ and say the output is simply $z_2$.
```

```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    index_affine_to_hom
    index_hom_to_affine
    change_chart
    hom_to_affine
    affine_to_hom
    fs_metric
    fs_potential
```

### Local coordinates
Going from affine coordinates to coordinates on the variety $X$ itself we must eliminate one additional redundant entry.
Given coordinate values $z_1^{(p)}, \ldots, z_{d}^{(p)}$ we can recover $z_0^{(p)}$ by solving the defining equation.
The simplest way to pick coordinates is by choosing one coordinate index that is to be considered "dependent" on the other values (c.f. implicit function theorem).
All other coordinate entries are kept.

As a particular example, consider a $(d+1) \times (d+1)$ matrix $g$ denoting a metric in the ambient projective space.
We now want to compute the pullback of this to the variety.
We have
$$
g^X_{i\bar{\jmath}} = \frac{\partial z_p^{k}}{\partial z_p^{i}} \frac{\partial \bar{z}_p^{\bar{l}}}{\partial \bar{z}_p^{\bar{\jmath}}} g_{k\bar{l}} \,,
$$
where $i$ and $\bar{\jmath}$ only range over $d$ values corresponding to the choice of dependent entry.
If $z_p^{m}$, i.e. index $m$, is chosen as dependent variable, only $\partial z_p^{m}/\partial z_p^i$ is non-trivial (all others being either one or zero).
These Jacobians can be computed directly (and automatically) from the defining equation:
$$
\frac{dz_{m}}{dz_j} = - \frac{dQ}{dz_j} \left( \frac{Q}{dz_{m}} \right)^{-1} \,.
$$

```{note}
Just as the affine patch can be chosen automatically by minimizing the numerical values that appear, we can also choose the dependent coordinate automatically by minimizing the value of the holomorphic top form $\Omega$ introduced later.
```

```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    complex_hessian
    induced_metric
    jacobian_embed
```

## Varieties
At present, varieties with a single defining equation on one complex projective space are defined.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    VarietySingle
```
Internally, they primarily contain a single homogeneous polynomial object which represents the defining equation.
A lot of functionality thus derives from the (homogeneous) polynomial objects.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    polynomial.Poly
    polynomial.HomPoly
```
Practically, it may often be unnecessary to interact directly with the polynomial objects.

Instead of passing a polynomial object, varieties can be created from a sympy expression or even a string representing the defining equation.
``Fermat`` and ``Dwork`` family varieties are also predefined for convenience.

```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::
    
    VarietySingle.from_sympy
    Fermat
    Dwork
```

Another important function that is defined by varieties is sampling.
The method we use here is based on finding intersections with a line in projective space and is explained in more detail in the section on {doc}`sampling`.

If we denote the defining homogeneous polynomial as $Q$, the points on the variety are defined by the equation $Q(z) = 0$. 
Two things we can immediately do is to evaluate $Q$ on points of the ambient projective space (either in homogeneous coordinates or affine coordinates given a patch),
and compute its derivative $\partial Q / \partial z$.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    VarietySingle.defining
    VarietySingle.grad_defining
```
Via the Jacobian of the embedding mentioned above, we can thus define the pullback from ambient space to local variety coordinates.
As it involves dividing by the gradient of the defining equation with respect to the dependent coordinate, we can define the "optimal" dependent coordinate as the one maximizing this.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    VarietySingle.best_dependent
    VarietySingle.jacobian_embed
    VarietySingle.induced_metric
    VarietySingle.induced_fs
```
Given local variety coordinates, we can also solve for the dependent coordinate using the defining equation.
```{eval-rst}
.. currentmodule:: cyjax

.. autosummary::

    VarietySingle.solve_defining
```
