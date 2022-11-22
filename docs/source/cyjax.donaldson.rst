cyjax.donaldson
===============

.. currentmodule:: cyjax.donaldson

.. automodule:: cyjax.donaldson

The core functionality is exposed an algebraic metric object which contains a variety and a choice of line bundle sections.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    AlgebraicMetric
    LBSections
    MonomialBasisFull
    MonomialBasisReduced

Some functionality is also available as dedicated functions.

.. autosummary::
    :toctree: _autosummary

    monomial_basis_indices
    monomial_basis
    monomial_basis_size
    reduced_monomial_basis

Internally, the computation of geometric quantities is handled by a computational graph class.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    GeometricObjects
