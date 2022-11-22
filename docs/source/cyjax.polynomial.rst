cyjax.polynomial
================

.. currentmodule:: cyjax.polynomial

.. automodule:: cyjax.polynomial


General polynomials
-------------------

The base functionality is implemented in the :class:`Poly` class:

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    Poly


It can be constructed either based on a (sympy) expression of the equation, or given matrices of powers and coefficients:

.. autosummary::

    Poly.from_sympy
    Poly.from_coeffs_and_powers
    Poly.to_coeffs_and_powers


Several helper functions are implemented that are used inside :class:`Poly`:

.. autosummary::
    :toctree: _autosummary

    compute_monomials
    split_subscript
    merge_subscript
    any_symbolic


Extract coefficients from multivariate polynomial
-------------------------------------------------

.. autosummary::
    :toctree: _autosummary

    univariate_coefficients


Homogeneous polynomials
-----------------------
Based on the :class:`Poly` class, a homogeneous polynomial class is defined which itself inherits from :class:`Poly`.

.. autosummary::
    :template: custom-class-template.rst
    :toctree: _autosummary

    HomPoly

The Dwork family and the Fermat polynomial are given for convenience.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    DworkPoly
    FermatPoly