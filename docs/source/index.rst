Calabi-Yau metrics with JAX
===========================

**CYJAX** is a python library for numerically approximating Calabi-Yau metrics
using machine learning implemented with the `JAX library <https://github.com/google/jax>`_.
It is meant to be accessible both as a top-level library as well as a toolkit of modular functions.
As of now, this implementation is limited to varieties given by a single defining equation on one complex projective space.
A generalization to a wider class of cases is planned.

Good places to start are the introduction and the tutorial notebooks listed below.
More background can also be found in this `paper <https://inspirehep.net/literature/1835403>`_.
Some background knowledge on `JAX <https://github.com/google/jax>`_ and
`Flax <https://github.com/google/flax>`_ may be helpful.

The introduction gives a summary of the mathematical context and aim of the library, which serves to give a broad overview to the structure and code of the library.
The tutorials show how to use the library on a code level and give several examples.

Conventions
-----------

- Generally, when the patch is `None` the coordinates are assumed to be homogeneous coordinates.
  Otherwise, the patch gives the index for the affine patch.
- The index of the affine patch is given in terms of the homogeneous coordinates.
- The dependent coordinate index (for which we can solve using the defining equation) is given in terms of the **affine** coordinate vector.
  Specifically, this means that, numerically, the dependent and the patch index may have the same value but do not refer to the same coordinate index.
- Input variables of polynomials should have integer subscripts as in :math:`z_0, z_1, z_2`.
  Parameters can use any valid sympy symbol expression including non-numerical subscripts.


Table of contents
-----------------

.. toctree::
    :maxdepth: 1
    :caption: Introduction

    intro/varieties
    intro/sampling
    intro/algebraic_ansatz
    intro/machine_learning


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    notebooks/projective_coordinates
    notebooks/varieties
    notebooks/sampling
    notebooks/geometry
    notebooks/donaldson
    notebooks/machine_learning

.. toctree::
    :maxdepth: 3
    :caption: API Documentation

    cyjax


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
