r"""Methods for manipulating polynomials.

The base class ``Poly`` builds on top of
`sympy.Poly <https://docs.sympy.org/latest/modules/polys/index.html>`_
and adds numerically efficient evaluation. In addition, it groups (scalar)
input variables into groups which represent some complex vector space
(later :math:`\mathbb{P}^n`).
For example, :math:`z_0, z_1, z_2, z_3 \in \mathbb{C}` are grouped into a
single vectorial input variable :math:`z`. Besides input variables,
the class also tracks a list of scalar parameters which are passed
together as a single array when evaluating the polynomial.

As a subclass of ``Poly``, a homogeneous polynomial class ``HomPoly`` is given,
which can be evaluated efficiently given affine coordinates.
"""
from __future__ import annotations

import chex
import sympy
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Sequence
from functools import partial, reduce, wraps
from collections import defaultdict
from chex import Array, Numeric
from sympy.printing import latex as _sympy_latex
from sympy.printing.conventions import split_super_sub
from inspect import Parameter, Signature, signature
from itertools import product as _cartesian
from .util import prod
from jax_autovmap import auto_vmap

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Sequence


def latex(obj):
    """Generate latex for string-symbol or sympy object."""
    if isinstance(obj, str):
        obj = sympy.Symbol(obj)
    return _sympy_latex(obj)


def split_subscript(var: Union[sympy.Symbol, str]) -> tuple[str, Optional[int]]:
    """Split variable into main name and integer subscript (or None).

    Both z1 and z_1 are permitted syntax and are treated equivalently.
    Internally, they are replaced by z1 (see :func:`merge_subscript`).
    Superscripts are not supported.
    """
    if not isinstance(var, str):
        var = str(var)

    name, sup, sub = split_super_sub(var)

    if len(sup) > 0:
        warnings.warn(f'Variable {var} has a superscript, which is currently '
                      f'not supported. It will be discarded.')

    if len(sub) == 0:
        return name, None
    if len(sub) > 1:
        raise RuntimeError(
            f'Variable {var} appears to have multiple subscripts,'
            f'which is currently not supported')

    sub = sub[0].strip('{').strip('}')
    try:
        return name, int(sub)
    except ValueError:
        raise ValueError(
            f'Variable subscripts should be integers, but got "{sub}"')


def merge_subscript(name: str, sub: int):
    """Join variable name and subscript to sympy symbol.

    For brevity in the printed expression, the style used is z1 i.e.
    without an underscore.
    """
    return sympy.Symbol(f'{name}{sub}')


def _union_indices(left: Sequence[int], right: Sequence[int]) -> Sequence[int]:
    ind = []
    added = set()
    iter_left = (i for i in left)
    iter_right = (i for i in right)
    nl = nr = None
    while True:
        try:
            nl = next(iter_left) if nl is None else nl
            nr = next(iter_right) if nr is None else nr
        except StopIteration:
            break
        nl, nr, add = (None, nr, nl) if nl < nr else (nl, None, nr)
        if add not in added:
            ind.append(add)
            added.add(add)
    # add pending next left / next right
    for i in (nl, nr):
        if i is not None and i not in added:
            ind.append(i)
            added.add(i)
    # works because either iter_left or iter_right is already exhausted
    ind.extend(i for i in iter_left if i not in added)
    ind.extend(i for i in iter_right if i not in added)
    return ind


def any_symbolic(*args: Union[np.ndarray, Array, sympy.Symbol, int, float, complex]):
    """Check if any of the arguments are symbolic instead of numerical."""
    return any(
        v is not None and (
            isinstance(v, (sympy.Expr, str))
            or (isinstance(v, (jnp.ndarray, np.ndarray))
                and v.dtype.kind in {'S', 'U', 'O'}))
        for v in args)


def _sort_var_args(variable_names, parameters, var_args, var_kwargs):
    """Get input variables ordered by variable name.

    In the input when evaluating polynomials, variables and params can be
    passed either positionally or by keyword. This function unpacks
    the received arguments into the sequence of variables and the parameters.
    """
    params = None
    if 'params' in var_kwargs:
        params = var_kwargs.pop('params')
    elif len(var_args) == len(variable_names) + 1:
        *var_args, params = var_args
    elif len(var_args) > len(variable_names) + 1:
        raise RuntimeError(
            f'Only variables and params can be passed positionally, '
            f'which in total are {len(variable_names)+1} arguments, '
            f'but got {len(var_args)} arguments instead.')

    variables = list(var_args[:len(variable_names)])
    for i in range(len(variables), len(variable_names)):
        variables.append(var_kwargs[variable_names[i]])

    assert params is None or len(params) == len(parameters), \
        f'Expected {len(parameters)} parameters but got {len(params)}.'

    return variables, params


def _sympy_eval_num(val):
    """Get numerical value for sympy expression of a number."""
    if val.func == sympy.Integer:
        return int(val)
    elif val.func == sympy.core.numbers.One:
        return 1
    elif val.func == sympy.core.numbers.Zero:
        return 0
    elif val.func == sympy.core.numbers.ImaginaryUnit:
        return 1j
    else:
        return float(val)


def _eval_sympy_poly(p, var_names, index_reorder, par_index):
    """Recursively return function evaluating the sympy expression."""
    if p.func == sympy.Symbol:
        name, index = split_subscript(p)
        if name in var_names:
            reo = index_reorder[name]
            if reo is None:
                return lambda var, par: var[name]
            else:
                i = reo[index]
                return lambda var, par: var[name][i]
        else:
            index = par_index[p]
            return lambda var, par: par[index]

    elif issubclass(p.func, sympy.Number):
        num = _sympy_eval_num(p)
        return lambda var, par: num

    elif p.func in [sympy.Add, sympy.Mul]:
        fns = []
        for arg in p.args:
            fn = _eval_sympy_poly(arg, var_names, index_reorder, par_index)
            fns.append(fn)
        op = sum if p.func == sympy.Add else prod
        return lambda z, var: op(f(z, var) for f in fns)

    elif p.func == sympy.Pow:
        fn1 = _eval_sympy_poly(p.args[0], var_names, index_reorder, par_index)
        fn2 = _eval_sympy_poly(p.args[1], var_names, index_reorder, par_index)
        return lambda z, var: fn1(z, var) ** fn2(z, var)

    else:
        raise RuntimeError('Encountered unknown operation: %s.' % p.func)


def _replace_float_ints(expr: sympy.Expr) -> sympy.Expr:
    """Replace all integer-valued floats in Expr with integers."""
    reps = dict()
    e = expr.replace(
        lambda x: x.is_Float and x == int(x),
        lambda x: reps.setdefault(int(x), sympy.Dummy()))
    return e.xreplace({v: k for k, v in reps.items()})


@partial(jax.jit, static_argnums=3)
def _compute_mon_scalar(arr, power, product, log):
    out = jax.lax.cond(
        power == 0,
        lambda args: args[2],
        lambda args: (args[2] + args[0] * args[1] if log else
                      args[2] * args[0] ** args[1]),
        (arr, power, product))
    return out


@partial(jax.vmap, in_axes=(None, 0, 0, None))
def _compute_mon(arr, power, product, log):
    # here arr is a single array/variable
    # and power a matching sequence of powers
    product, _ = jax.lax.scan(
        (lambda product, ap: (_compute_mon_scalar(*ap, product, log), None)),
        product,
        (arr, power))
    return product


def compute_monomials(
        values: Sequence[Array, ...],
        coeffs: Array,
        powers: Sequence[Sequence[Sequence[int]]], *,
        opt=True, log=True, symbolic=False):
    """Compute monomial terms of a polynomial.

    Setting log=True appears to lead to slightly faster computation.
    Using opt=True only leads to a speedup if there are powers == 0.
    In that case, the value can be skipped and replaced by 1 without
    computation.

    Args:
        values: Tuple of values (variables). Each can have rank 0 (for scalars)
            or rank 2 (for vector variables).
        coeffs: Sequence (1D) of coefficients for each monomial.
        powers: Tuple of powers for each variable.
            The first index ranges over the included variables,
            the second over the monomials,
            and the third over the indices of the variables.
        log: Whether to use exp(power * log(var)) for computation.
        opt: Whether to use an evaluation structure that avoids computations
            where power==0. This is needed for numeric reasons when ``log``
            is true, thus the argument is ignored in that case.
        symbolic: Indicates the inputs are symbolic and not purely numerical.
            If so, both ``log`` and ``opt`` are ignored.

    Returns:
        Array of monomials evaluated for given values.
    """
    if symbolic:
        values = [np.atleast_1d(v) if v is not None else v for v in values]
        products = np.ones(coeffs.shape)
        for v, p in zip(values, powers):
            if v is None or p is None:
                assert p is None or len(p) == 0, \
                    f'Cannot raise variable with value None to power {p}.'
                continue
            products = products * np.prod(v ** np.array(p), axis=-1)
        return np.asarray(coeffs) * products
    else:
        values = [jnp.atleast_1d(v) if v is not None else v for v in values]

    init = jnp.zeros if log else jnp.ones
    dtype = jnp.result_type(*values, coeffs)
    products = init(coeffs.shape, dtype=dtype)

    for v, p in zip(values, powers):
        # Permit value None if power is None. This is useful when a parameter
        # does not appear in the polynomial but is still included as
        # a parameter for generality.
        if v is None or p is None:
            assert p is None or len(p) == 0, \
                f'Cannot raise variable with value None to power {p}.'
            continue
        if log:
            v = jnp.log(v)
            if dtype not in (jnp.complex64, jnp.complex128):
                warnings.warn(
                    'Using the log-method to evaluate polynomials '
                    'with real input values may lead to NaN values '
                    'if any of the values is negative.')
        if log or opt:
            # scan over monomials
            products = _compute_mon(v, jnp.asarray(p), products, log)
        else:
            products *= jnp.prod(v ** jnp.asarray(p), axis=-1)

    return coeffs * jnp.exp(products) if log else coeffs * products


class Poly:
    #: Names of variables of the polynomial. Each can be scalar or vectorial.
    variable_names: Sequence[str]
    #: Indices for each variable or None (if scalar).
    variable_indices: Sequence[Optional[Sequence[int]]]
    #: Parameters of the polynomial as sequence of sympy symbols.
    #: Each parameter is a scalar and the ``params`` input when evaluating
    #: the polynomial is 1D, specifying their values in the order given here.
    parameters: Sequence[sympy.Symbol]  # parameters do not appear with powers

    # allow subclasses to generate this on the fly (via sympy_poly property)
    _sympy_poly: sympy.Poly = None

    def __init__(self,
                 sympy_poly: Optional[sympy.Poly],
                 variable_names: Sequence[str],
                 parameters: Sequence[sympy.Symbol],
                 variable_indices: Sequence[Optional[Sequence[int]]]):
        """Parametrized polynomial class.

        Instead of using the default constructor, it is typically safer
        and simpler to use one of the following two class methods:

        - :func:`Poly.from_sympy`
        - :func:`Poly.from_coeffs_and_powers`
        """
        self.variable_names = variable_names
        self.variable_indices = variable_indices
        self.parameters = parameters
        self._sympy_poly = sympy_poly

        # create call signature
        orig = signature(self.__call__).parameters  # original parameters
        par = [
            Parameter(
                name, Parameter.POSITIONAL_OR_KEYWORD,
                default=orig[name].default if name in orig else Parameter.empty)
            for name in [*self.variable_names, 'params']]

        extra = self._extra_args()
        for name in extra:
            p = Parameter(
                name, Parameter.KEYWORD_ONLY,
                default=orig[name].default if name in orig else Parameter.empty)
            par.append(p)

        self.__signature__ = Signature(par)

    @property
    def domain(self) -> sympy.polys.Domain:
        """Complex domain of the polynomial.

        Polynomials here are defined over the field or complex numbers.
        The ``Domain`` object additionally tracks the parameters of the
        polynomial.
        """
        if len(self.parameters) == 0:
            return sympy.CC
        return sympy.CC[self.parameters]

    @property
    def sympy_poly(self) -> sympy.Poly:
        """Representation of the polynomial as ``sympy.Poly`` object.

        Note that all variables are unpacked to be scalars in this form.
        """
        return sympy.Poly(self(*self.all_symbols()), domain=self.domain) \
            if self._sympy_poly is None else self._sympy_poly

    @property
    def parameter_position(self) -> dict[sympy.Symbol, int]:
        """Mapping between parameter and its index in the ordering.

        The ordering is the one the input is expected to be in when
        evaluating the polynomial.
        """
        return {p: i for i, p in enumerate(self.parameters)}

    @property
    def variable_position(self) -> dict[str, int]:
        """Mapping between variable name and its index in the ordering.

        The ordering is the one the inputs are expected to be in,
        if given positionally, when evaluating the polynomial.
        """
        return {p: i for i, p in enumerate(self.variable_names)}

    @property
    def variable_index_dict(self) -> dict[str, dict[int, int]]:
        """Mappings between positional index and symbolic index of variables.

        To represent e.g. the affine coordinates ``[z0, z2, z3]`` where
        ``z1=1`` is omitted, the symbolic indices ``[0, 2, 3]`` are stored.
        For each variable name (e.g. ``'z'``) the output maps between
        the numeric index of the input variable to the symbolic indices.
        In the example, the output would be ``{'z': {0: 0, 1: 2, 2: 3}}``.
        """
        return {
            name: None if ind is None else {
                index: i for i, index in enumerate(ind)}
            for name, ind in zip(self.variable_names, self.variable_indices)}

    def _extra_args(self) -> dict[str, tuple[bool, Optional[int]]]:
        """Dictionary giving tuple(need static jit, rank) for new arguments.

        To enable generation of the efficient numerical evaluation function,
        need to know about new variables added to __call__ and whether they
        must be marked static or if not, what their base rank for auto_vmap is.
        """
        return dict()

    def transform_eval(self, **partial_kwargs):
        """Generate an efficient numerical evaluation function.

        In contrast to calling the polynomial object directly,
        the inputs to the generated functions can have any leading
        batch dimensions but must be purely numerical.
        """
        dims = {
            v: 0 if idc is None else 1
            for v, idc in zip(self.variable_names, self.variable_indices)}
        dims['params'] = 1

        static = self._extra_args()
        for v, (_, dim) in static.items():
            if dim is not None:
                dims[v] = dim
        names = [v for v, (s, _) in static.items()
                 if s and v not in partial_kwargs]

        fn = partial(auto_vmap(**dims)(self), **partial_kwargs)
        return jax.jit(fn, static_argnames=names)

    @classmethod
    def from_poly(cls, poly: Poly) -> Poly:
        """Initializer with consistent signature for all Poly subclasses.

        Motivation: Allows conversion to subclasses and also allows
        easy inheritance of from_sympy and other construction methods.
        These will first construct a ``Poly`` object which is then
        converted to the right class using this method. In subclasses,
        it should be overridden if the __init__ signature is changed.
        Otherwise, the default implementation will work.
        """
        return cls(poly.sympy_poly, poly.variable_names,
                   poly.parameters, poly.variable_indices)

    @classmethod
    def from_sympy(
            cls,
            poly: Union[str, sympy.Expr, sympy.Poly],
            variable_names: Union[Sequence[str], Sequence[NDArray[sympy.Symbol]]] = None,
            parameters: Sequence[Union[sympy.Symbol, str]] = None,
            variable_dim: Sequence[int] = None,
            variable_indices: Sequence[Optional[Sequence[int]]] = None):
        """Construct a :class:`Poly` object from a sympy/string expression.

        Variables should either have integer subscripts or no subscripts.

        The function tries to automatically detect variables and parameters
        appearing in the expression. To make sure this behaves properly,
        multiple things can be (optionally) specified:

        - The variable names. If given, every other symbol in the expression
          is interpreted as a parameter.
        - The parameters. If given, every other symbol in the expression is
          interpreted as a variable.
        - The variable dimensions (must also give variable names in this case).
          If, for example ``z0`` and ``z2`` appear in the expression
          but not ``z1``, the latter index is by default *not* assumed
          to be part of the ``z`` variable. To be sure it is, the dimension
          can be specified as ``3`` (or higher).
        - The integer indices for each variable (or None if scalar).

        If either the parameters or the variable names are given but not both,
        all encountered symbols that do not match the given kind are assigned
        to the other. If both are given, all encountered symbols must be
        accounted for.

        Args:
            poly: Either a string or a sympy expression for the equation
                of the polynomial.
            variable_names: Sequence of strings giving the names of the
                input variables of the polynomial.
                Alternatively, a sequence of numpy arrays containing sympy
                symbols can be passed. This will fully specify both the
                variable names and their indices (given by the subscripts
                of the symbols).
            parameters: Sequence of strings or sympy symbols specifying
                the (scalar) parameters of the polynomial.
            variable_dim: Integer dimension for each variable.
                If given, the indices for the variable are set to be
                ``0, 1, ..., dim-1``.
            variable_indices: Sequence of integer indices
                (or ``None`` if scalar) for each variable.

        Returns:
            A :class:`Poly` object representing the given expression as
            a parametrized polynomial.
        """

        have_variables = variable_names is not None
        have_indices = variable_indices is not None
        have_parameters = parameters is not None
        if not have_indices and variable_dim is not None:
            have_indices = True
            variable_indices = [list(range(d)) for d in variable_dim]
        assert have_variables or not have_indices, \
            'If indices are provided, variable names must also be given.'

        # make sure all are symbols (not str)
        if have_parameters:
            parameters = [sympy.Symbol(p) if isinstance(p, str) else p
                          for p in parameters]
        if have_variables:
            all_str = all(isinstance(vn, str) for vn in variable_names)
            all_sym = all(isinstance(vn, np.ndarray) for vn in variable_names)
            assert all_sym ^ all_str, \
                'Variable_names must either all be strings or all ' \
                'numpy arrays of variables.'
            if all_sym:
                have_indices = True
                flat_vars = [v.flatten() for v in variable_names]
                variable_names = [
                    split_subscript(var[0])[0] if len(var) != 0 else 'dummy'
                    for var in flat_vars]
                variable_indices = [
                    None if len(var) == 1 and split_subscript(var[0])[1] is None
                    else [split_subscript(v)[1] for v in var]
                    for name, var in zip(variable_names, flat_vars)]

        # sympy.Poly object specifies which values are constants/variables
        if isinstance(poly, sympy.Poly):
            poly_expr = poly.expr
            if not have_variables:
                have_variables = True
                variable_names = list({split_subscript(g)[0] for g in poly.gens})
            if parameters is None and hasattr(poly.domain, 'symbols'):
                have_parameters = True
                parameters = parameters or poly.domain.symbols
        elif isinstance(poly, sympy.Expr):
            poly_expr = poly
        else:
            poly_expr = sympy.parse_expr(poly)

        # if parameters not known, will append them to a new list
        # -> replace None with empty list
        parameters = [] if parameters is None else parameters
        variable_names = [] if variable_names is None else variable_names
        variable_indices = [None] * len(variable_names) \
            if variable_indices is None else variable_indices

        var_appeared = [False] * len(variable_names)
        # clean variable names (use a1 instead of a_1 for consistency)
        rename = dict()
        # index of variable name in list of variables
        var_name_index = {n: i for i, n in enumerate(variable_names)}

        for v in poly_expr.free_symbols:
            assert isinstance(v, sympy.Symbol), \
                f'Encountered free symbol {v} that is not of type Symbol.'
            if v in parameters:
                continue
            name, index = split_subscript(v)

            if have_variables and name not in variable_names:
                # in this case must be a parameter
                if have_parameters:
                    assert v in parameters, \
                        f'Parameter {v} appears in polynomial expression ' \
                        f'but does not appear in list of parameters. {v} not in {parameters}'
                else:
                    parameters.append(v)
                continue

            try:
                var_index = var_name_index[name]
            except KeyError:
                if have_variables:
                    raise ValueError(
                        f'Variable {v} appears in polynomial expression '
                        f'but does not appear in list of variable names.')
                else:
                    # found a new variable
                    variable_names.append(name)
                    variable_indices.append(None if index is None else [index])
                    var_appeared.append(True)
                    continue

            if index is None:
                # If it appeared before, it must have appeared with index
                # since the values in poly.gens are not repeated.
                assert variable_indices[var_index] is None, \
                    f'Variable "{name}" appeared both with and ' \
                    f'without subscript.'
                continue
            elif have_indices:
                # Indices already available, only make sure it is not missing.
                assert index in variable_indices[var_index], \
                    f'Variable index {index} missing in specified list of ' \
                    f'indices for variable {v}.'
            elif variable_indices[var_index] is None:
                # Either no indices collected yet, or appeared without before
                assert not var_appeared[var_index], \
                    f'Variable "{name}" appeared both with and ' \
                    f'without subscript.'
                variable_indices[var_index] = [index]
            else:
                # Add to collected indices.
                variable_indices[var_index].append(index)
                var_appeared[var_index] = True

            rename[v] = merge_subscript(name, index)
        if not have_indices:
            variable_indices = [ind if ind is None else list(sorted(ind))
                                for ind in variable_indices]
        if not have_parameters:
            # sympy yields unpredictable order
            parameters = list(sorted(parameters, key=str))
        # temporarily init Poly object
        poly_obj = Poly(None, variable_names, parameters, variable_indices)
        variables = np.array(poly_obj.all_symbols(False, False))
        poly_obj._sympy_poly = sympy.Poly(
            poly_expr.subs(rename), *variables, domain=poly_obj.domain)
        # return a new object of the right type (for subclasses)
        return cls.from_poly(poly_obj)

    def all_symbols(self, as_arrays=True, include_params=True) -> list[Union[sympy.Symbol, np.ndarray]]:
        """All symbols appearing as inputs to the polynomial.

        Args:
            as_arrays: If true, group (non-scalar) variables with indices
                into a single numpy array. Otherwise, one symbol is
                added to the output for each index.
                Similarly, all parameters are put into a single numpy array
                if true.
            include_params:
                If true, includes the parameters of the polynomial.
        Returns:
            A list, either containing sympy symbols representing the
            inputs of the polynomial, or numpy arrays of symbols which are
            grouped in the same way as the numerical inputs are when
            evaluating the polynomial.
        """
        symbols = []
        for name, indices in zip(self.variable_names, self.variable_indices):
            if indices is None:
                sym = sympy.Symbol(name)
                symbols.append(np.array(sym) if as_arrays else sym)
                continue
            var = (merge_subscript(name, i) for i in indices)
            if as_arrays:
                symbols.append(np.array(list(var)))
            else:
                symbols.extend(var)
        if include_params:
            if as_arrays:
                symbols.append(np.array(self.parameters))
            else:
                symbols.extend(self.parameters)
        return symbols

    @classmethod
    def from_coeffs_and_powers(
            cls,
            coeffs: Sequence[Numeric],
            powers: Sequence[Sequence[Sequence[int]]],
            coeffs_params: Sequence[Sequence[Union[sympy.Symbol, str]]] = None,
            variable_names: Sequence[str] = None,
            variable_indices: Sequence[Optional[Sequence[int]]] = None,
            parameters: Sequence[Union[sympy.Symbol, str]] = None):
        r"""Construct :class:`Poly` object from arrays of powers and coeffs.

        The polynomial is constructed from the inputs as follows:

        .. math::
            \sum_c \mathrm{coeffs}_c
                \prod_p \mathrm{coeffs\_params}_{cp}
                \prod_{v}
                    \prod_{i \in \mathrm{variable\_indices}_v}
                        \mathrm{variable\_names[v]}_i^{\mathrm{powers}_{vci}}

        By default, variable indices are derived as ranges with length
        given by the shape of the powers.
        If no variable names are given, they are set to ``a, b, c`` etc.
        Parameters can be automatically extracted from ``coeffs_params``.

        Args:
            coeffs: Sequence of numerical coefficients of monomials in the
                full polynomial expression.
            powers: For each variable, gives the powers in each monomial
                and for each index of the variable (in that order of indexing).
            coeffs_params: Symbolic coefficients of monomial terms.
            variable_names: Names of variables. If not given, pick letters
                from the alphabet.
            variable_indices: Indices of variables. If not given, can
                derive dimensions (number of indices) from the shape of the
                corresponding power array.
            parameters:
                Sequence of parameters. If given, sets the order of parameters
                that is expected when evaluating the polynomial.
                Important to explicitly specify if not all parameters actually
                appear in the polynomial.
        Returns:
            A :class:`Poly` object representing the polynomial.
        See Also:
            :func:`compute_monomials`,
            :func:`Poly.to_coeffs_and_powers`
        Notes:
            Internally, the constructed :class:`Poly` object
            still uses a sympy expression.
            If desired, a subclass could be defined which internally
            maintains the representation in terms of power and coefficient arrays,
            using :func:`compute_monomials` for efficient numerical evaluation.
        """
        if coeffs_params is None:
            coeffs_params = [[]] * len(coeffs)
        else:
            coeffs_params = [
                [sympy.Symbol(p) if isinstance(p, str) else p for p in p_list]
                for p_list in coeffs_params]

        if variable_names is None:
            variable_names = [
                chr(ord('a') + i) for i in range(len(powers))]
        if variable_indices is None:
            variable_indices = [range(len(pws[0])) for pws in powers]
        if parameters is None:
            parameters = list(reduce(set.union, map(set, coeffs_params)))
        else:
            parameters = [sympy.Symbol(p) if isinstance(p, str) else p
                          for p in parameters]

        # construct sympy polynomial
        poly = Poly(None, variable_names, parameters, variable_indices)
        par_index = poly.parameter_position

        coeffs = np.array([
            coeff * prod(parameters[par_index[p]] for p in par_coeff)
            for coeff, par_coeff in zip(coeffs, coeffs_params)])

        sympy_poly = np.sum(compute_monomials(
            poly.all_symbols(True, False), coeffs, powers, symbolic=True))

        poly._sympy_poly = sympy.Poly(sympy_poly, domain=poly.domain)
        return cls.from_poly(poly)

    def to_coeffs_and_powers(self) \
            -> tuple[Sequence[Numeric],
                     Sequence[Sequence[sympy.Symbol]],
                     Sequence[Sequence[Sequence[int]]]]:
        """Extract powers of variables and lists of coefficients.

        This is effectively the inverse of :func:`Poly.from_coeffs_and_powers`.
        """
        sympy_poly = self.sympy_poly

        coeffs = list()
        coeffs_params = list()
        powers = tuple(list() for _ in self.variable_names)

        all_vars = [v.flatten() for v in self.all_symbols(True, False)]
        var_poly_index = {
            v: sympy_poly.gens.index(v) if v in sympy_poly.gens else None
            for v in np.concatenate(all_vars)}
        for p in self.parameters:
            try:
                var_poly_index[p] = sympy_poly.gens.index(p)
            except ValueError:
                var_poly_index[p] = None

        def _add_term(pows, coeff):
            coeffs.append(1)
            coeffs_params.append([])

            def _add_coeff(p):
                if isinstance(p, sympy.Symbol):
                    assert p in self.parameters, \
                        f'Encountered parameter {p} which ' \
                        f'is not in list of parameters.'
                    coeffs_params[-1].append(p)
                else:
                    # assume only one numeric value in coeff.args
                    coeffs[-1] = _sympy_eval_num(p)

            if coeff.func == sympy.Mul:
                for c in coeff.args:
                    _add_coeff(c)
            else:
                _add_coeff(coeff)

            for var, var_pows in zip(all_vars, powers):
                power_entry = []
                var_pows.append(power_entry)
                for v in var:
                    index = var_poly_index[v]
                    if index is None:
                        power_entry.append(0)
                    else:
                        power_entry.append(pows[index])

        for pows, coeff in sympy_poly.terms():
            if coeff.func == sympy.Add:
                for c in coeff.args:
                    _add_term(pows, c)
            else:
                _add_term(pows, coeff)

        powers = tuple(
            None if all(all(p_i == 0 for p_i in p) for p in pows) else pows
            for pows in powers)

        return coeffs, coeffs_params, powers

    def _combine_metadata(self, other: Union[sympy.Symbol, Numeric, Poly]):
        """Combine poly data before applying operation (+, *, ...)."""
        params = list(self.parameters)
        var_names = list(self.variable_names)
        var_indices = list(self.variable_indices)
        if isinstance(other, (np.ndarray, jnp.ndarray)):
            other = other.item()  # must be a scalar

        if isinstance(other, sympy.Symbol) and other not in params:
            name, index = split_subscript(other)
            try:  # find in variables and add index
                var_index = var_names.index(name)
                indices = var_indices[var_index]
                # could allow case where only index is None, and introduce
                # this as a new parameter but might be more confusing
                assert not ((indices is None) ^ (index is None)), \
                    f'Variable {name} appears both with and without index ' \
                    f'in multiplied terms.'
                if index is not None:
                    var_indices[var_index] = _union_indices(indices, (index,))
                # This will make 'other' a variable. Else it would be parsed
                # as a parameter in the sympy.Poly.
                other = sympy.Poly(other, domain='CC')
            except ValueError:  # add to parameters
                params.append(other)

        if isinstance(other, Poly):
            for p in other.parameters:
                if p not in params:
                    params.append(p)
            for name, indices in zip(
                    other.variable_names, other.variable_indices):
                try:  # try to merge indices with existing variable
                    var_index = var_names.index(name)
                    ind = var_indices[var_index]
                    assert not ((ind is None) ^ (indices is None)), \
                        f'Variable {name} appears both with and without ' \
                        f'index in multiplied terms.'
                    if indices is not None:
                        var_indices[var_index] = _union_indices(ind, indices)
                except ValueError:  # new variable
                    var_names.append(name)
                    var_indices.append(indices)
        return Poly(None, var_names, params, var_indices), other

    def diff(self, var: Union[sympy.Symbol, str]) -> sympy.Poly:
        """Differentiate with respect to a variable using sympy."""
        sympy_poly = self.sympy_poly.diff(var)
        poly = Poly(sympy_poly, self.variable_names,
                    self.parameters, self.variable_indices)
        return self.from_poly(poly)

    def __mul__(self, other: Union[sympy.Symbol, Numeric, Poly]) -> Poly:
        """Multiply two polynomials."""
        out, other = self._combine_metadata(other)
        if isinstance(other, Poly):
            out._sympy_poly = self.sympy_poly * other.sympy_poly
        elif isinstance(other, sympy.Poly):
            out._sympy_poly = self.sympy_poly * other
        else:
            out._sympy_poly = self.sympy_poly * np.array(other)
        return out

    def __add__(self, other: Union[sympy.Symbol, Numeric, Poly]) -> Poly:
        """Add two polynomials."""
        out, other = self._combine_metadata(other)
        if isinstance(other, Poly):
            out._sympy_poly = self.sympy_poly + other.sympy_poly
        elif isinstance(other, sympy.Poly):
            out._sympy_poly = self.sympy_poly + other
        else:
            out._sympy_poly = self.sympy_poly + np.array(other)
        return out

    def __sub__(self, other: Union[sympy.Symbol, Numeric, Poly]) -> Poly:
        """Subtract two polynomials."""
        out, other = self._combine_metadata(other)
        if isinstance(other, Poly):
            out._sympy_poly = self.sympy_poly - other.sympy_poly
        elif isinstance(other, sympy.Poly):
            out._sympy_poly = self.sympy_poly - other
        else:
            out._sympy_poly = self.sympy_poly - np.array(other)
        return out

    def __repr__(self):
        """String representation of polynomial."""
        type_name = type(self).__name__
        poly_repr = str(_replace_float_ints(self.sympy_poly.expr))
        variables = ', '.join([
            name if ind is None
            else '[' + ', '.join(f'{name}_{i}' for i in ind) + ']'
            for name, ind in zip(self.variable_names, self.variable_indices)])

        if not self.parameters:
            return f'{type_name}({variables}) = {poly_repr}'

        parameters = '[' + ', '.join(map(str, self.parameters)) + ']'
        return f'{type_name}({variables}, params={parameters}) ' \
               f'= {poly_repr}'

    def _repr_latex_(self):
        """Latex representation of polynomial."""
        poly_repr = latex(_replace_float_ints(self.sympy_poly.expr))
        type_name = r'\operatorname{' + type(self).__name__ + '}'
        variables = ','.join([
            latex(name) if ind is None
            else '[' + ','.join([latex(f'{name}_{i}') for i in ind]) + ']'
            for name, ind in zip(self.variable_names, self.variable_indices)])
        if not self.parameters:
            return fr'$\displaystyle {type_name}({variables}) = {poly_repr}$'

        parameters = '[' + ','.join([latex(p) for p in self.parameters]) + ']'
        return fr'$\displaystyle {type_name}' \
               fr'({variables}, \text{{params}}={parameters}) = {poly_repr}$'

    def __call__(self, *var_args, **var_kwargs) -> Union[sympy.Expr, np.ndarray]:
        """Evaluate the polynomial.

        Input variables and a single 1D ``params`` array/list can
        be passed either positionally or by keyword argument.

        Inputs can be symbolic or numerical. If any of the inputs is symbolic,
        the output is a sympy expression.
        Note that the input here must be a single set of values and cannot
        have batch dimensions.

        See Also:
            :func:`Poly.transform_eval` to generate a numerically efficient
            function which does allow inputs to have leading batch dimensions.
        """

        if len(var_args) == 0 and len(var_kwargs) == 0:
            return _replace_float_ints(self.sympy_poly.expr)

        var, params = _sort_var_args(
            self.variable_names, self.parameters, var_args, var_kwargs)
        if any_symbolic(params, *var):
            # for symbolic evaluation, use sympy_poly.subs({old: new})
            var_list = []  # collect variables
            for v, name, indices in zip(
                    var, self.variable_names, self.variable_indices):
                if indices is None:
                    var_list.append(v)
                else:
                    assert len(v) == len(indices), \
                        f'Expected size {len(indices)} for variable ' \
                        f'{name}, but got {len(v)} instead.'
                    var_list.extend(v)
            if params is not None:
                var_list.extend(params)
            subs = {s: v for s, v in zip(self.all_symbols(False), var_list)}
            return self.sympy_poly.expr.subs(subs)

        # Convert sympy to function which can be JIT compiled.
        fn = _eval_sympy_poly(self.sympy_poly.expr,
                              self.variable_names,
                              self.variable_index_dict,
                              self.parameter_position)

        var_dict = {name: v for name, v in zip(self.variable_names, var)}
        return fn(var_dict, params)


def univariate_coefficients(
        poly: Poly,
        var: Union[str, sympy.Symbol, NDArray[sympy.Symbol]]) \
        -> list[Poly]:
    """Polynomials for coefficients in each order for a single variable.

    Given a multivariate polynomial ``poly``, we single one variable
    (``var``) out and treat the others as constants.
    The coefficients of the single variate polynomial in ``var`` are then
    given by polynomials in the remaining "constant" variables.
    This function extracts the coefficients of the univariate polynomial
    in ``var`` in terms of polynomials in the remaining constant variables.

    Example:
        >>> poly = 'z1**2 * z2 * x**2 + 3 * shift * (1 + x**2)'
        >>> variables = ['x', 'z', 'shift']
        >>> poly = Poly.from_sympy(poly, variables)
        >>> coeff_polies = univariate_coefficients(poly, 'x')
        >>> len(coeff_polies)  # degree in x is 2 so we get 3 coefficients
        3
        >>> print(coeff_polies[0])  # At degree x^2
        Poly([z_1, z_2], shift) = 3*shift + z1**2*z2
        >>> print(coeff_polies[1])  # At degree x^1
        0.0
        >>> print(coeff_polies[2])  # At degree x^0
        Poly([z_1, z_2], shift) = 3*shift

    Args:
        poly: multivariate polynomial.
        var: variable of interest.
    Returns:
        List of numerical values of instances of :class:`Poly`,
        representing the coefficients of the polynomial in ``var``.
    """
    if isinstance(var, str):
        var = sympy.Symbol(var)
    elif isinstance(var, (np.ndarray, jnp.ndarray)):
        var = var.item()
    assert var not in poly.parameters, \
        'The variable by which to factor must not be a parameter.'

    # remove the chosen variable by which we factor
    all_vars = poly.all_symbols(True, False)
    consts = [v[v != var].reshape((-1,) * v.ndim)
              for v in all_vars if np.any(v != var)]

    sympy_poly = sympy.Poly(poly.sympy_poly, var)
    coeffs = defaultdict(lambda: jnp.zeros(()))
    for (p,), coeff in sympy_poly.terms():
        if coeff.is_number:
            coeffs[p] = np.array(coeff, dtype=complex)
        else:
            coeffs[p] = Poly.from_sympy(coeff, consts, poly.parameters)

    coeffs = [coeffs[i] for i in reversed(range(max(coeffs) + 1))]
    return coeffs


# To define varieties, we need to be able to compute the defining equation
# efficiently in terms of homogeneous *and* affine coordinates.
class HomPoly(Poly):
    # Attributes same as Poly
    variable_names: Sequence[str]
    variable_indices: Sequence[Optional[Sequence[int]]]
    parameters: Sequence[sympy.Symbol]

    _affine_poly: list[Poly]

    def __init__(self,
                 sympy_poly: Optional[sympy.Poly],
                 variable_names: Sequence[str],
                 parameters: Sequence[sympy.Symbol],
                 variable_indices: Sequence[Optional[Sequence[int]]]):
        """Homogeneous polynomial.

        Practically, this class allows the computation of the homogeneous
        polynomial itself in terms of homogeneous coordinates, as well as
        in terms of any set of affine coordinates.
        The latter are well-defined up to overall scaling.
        Specifically, the scale ambiguity is fixed by assuming the
        homogeneous coordinate omitted to get the affine coordinates was
        rescaled to the value 1.

        Just like :class:`Poly`, the class splits the inputs into one or more
        variables and parameters.
        """
        super().__init__(sympy_poly, variable_names, parameters, variable_indices)
        # store created affine polynomials
        total_patches = prod(len(idc) for idc in self.variable_indices)
        self._affine_poly = [None] * total_patches

    def unique_patch_index(self, patch: Union[int, Sequence[int]]) -> int:
        """Convert patch indices to single unique integer index.

        This is used internally to dispatch calls using affine coordinate
        inputs to the right affine polynomial.

        Example:
            If there are three variables with dimensions ``d1, d2, d3``, and
            we have patch indices ``i, j, k``, the unique integer index
            is computed as ``i + d1 * (j + d2 * k)``.
            Patch indices range from ``0`` to ``dimension-1``.
        """
        if jnp.isscalar(patch) or (
                isinstance(patch, jnp.ndarray) and patch.ndim == 0):
            patch = (patch,)

        index = 0
        multiplier = 1
        for p, ind in zip(patch, self.variable_indices):
            size = len(ind)
            index += p * multiplier
            multiplier *= size
        return index

    def affine_poly(self, patch: Union[int, Sequence[int]]) -> Poly:
        """Get affine polynomial for given patch.

        Important: The affine patch index is specified in terms of the actual
        numerical index into input variables. The indices used for
        pretty-printing is ignored. If these indices are consecutive and start
        at 0, there is no difference, however.

        Args:
            patch: Single or multiple patch indices. Each must be an integer
                between zero and the length of the variable (exclusively).

        Returns:
            A :class:`Poly` object representing the affine polynomial.
        """
        if jnp.isscalar(patch):
            patch = (patch,)

        patch_index = self.unique_patch_index(patch)
        p = self._affine_poly[patch_index]
        if p is None:
            subs = {merge_subscript(name, p): 1
                    for name, p in zip(self.variable_names, patch)}
            sympy_poly = self.sympy_poly.subs(subs)
            indices = [[i for i in ind if i != p]
                       for ind, p in zip(self.variable_indices, patch)]
            p = Poly(sympy_poly, self.variable_names, self.parameters, indices)
            self._affine_poly[patch_index] = p
            return p
        else:
            return p

    def _extra_args(self) -> dict[str, tuple[bool, Optional[int]]]:
        return {'patch': (False, 0)}  # not a static argument, rank 0

    def transform_eval(self, **partial_kwargs):
        # need to generate all affine polynomials to make it JIT compilable
        all_patches = _cartesian(*[
            range(len(idc)) for idc in self.variable_indices])
        all_patches = sorted(all_patches, key=lambda p: self.unique_patch_index(p))

        dims = {
            v: 0 if idc is None else 1
            for v, idc in zip(self.variable_names, self.variable_indices)}
        dims['params'] = 1
        static = self._extra_args()
        for v, (_, dim) in static.items():
            if dim is not None:
                dims[v] = dim

        sig = self.__signature__

        def branch_fn(args, kwargs, *, patch):
            return self.__call__(*args, **kwargs, patch=patch)

        branches = [partial(branch_fn, patch=patch) for patch in all_patches]

        @wraps(self.__call__)
        def eval_poly(*args, **kwargs):
            patch = kwargs.pop('patch')
            if patch is None:
                return self.__call__(*args, **kwargs, patch=None)
            else:
                patch_index = self.unique_patch_index(patch)
                return jax.lax.switch(patch_index, branches, args, kwargs)

        eval_poly.__signature__ = sig
        names = [v for v, (s, _) in static.items()
                 if s and v not in partial_kwargs]

        fn = partial(auto_vmap(**dims)(eval_poly), **partial_kwargs)
        return jax.jit(fn, static_argnames=names)

    def __call__(self, *var_args, patch: Union[int, Sequence[int]] = None, **var_kwargs):
        if patch is None:
            return super(HomPoly, self).__call__(*var_args, **var_kwargs)

        poly = self.affine_poly(patch)
        return poly(*var_args, **var_kwargs)


class DworkPoly(HomPoly):
    variable_names: Sequence[str]
    variable_indices: Sequence[Optional[Sequence[int]]]
    parameters: Sequence[sympy.Symbol]

    #: Polynomial degree.
    degree: int
    #: Pre-factor of the product term.
    factor: Numeric

    def __init__(self, poly_degree=5, factor: Union[bool, Numeric] = True):
        super(DworkPoly, self).__init__(
            None, ('z',), (sympy.Symbol('psi'),),
            [list(range(poly_degree))])
        self.degree = poly_degree
        if isinstance(factor, bool):
            self.factor = -5 if factor else 1
        else:
            self.factor = factor

    @classmethod
    def from_poly(cls, poly: Poly):
        return poly  # return generic polynomial (don't coerce to Dwork type)

    def __call__(self, z, params, patch: Union[int, Sequence[int]] = None):
        chex.assert_shape(z, (self.degree - (patch is not None),))
        numpy = np if any_symbolic(z, params) else jnp
        z, par = numpy.asarray(z), numpy.asarray(params)
        f = numpy.sum(z ** self.degree) + self.factor * par[0] * numpy.prod(z)
        return f if patch is None else 1 + f  # all patches symmetric


class FermatPoly(HomPoly):
    variable_names: Sequence[str]
    variable_indices: Sequence[Optional[Sequence[int]]]
    parameters: Sequence[sympy.Symbol]

    #: Polynomial degree.
    degree: int

    def __init__(self, poly_degree=5):
        super(FermatPoly, self).__init__(
            None, ('z',), (), [list(range(poly_degree))])
        self.degree = poly_degree

    @classmethod
    def from_poly(cls, poly: Poly):
        return poly  # return generic polynomial (don't coerce to Fermat type)

    def __call__(self, z, params=None, patch: Union[int, Sequence[int]] = None):
        chex.assert_shape(z, (self.degree - (patch is not None),))
        numpy = np if any_symbolic(z) else jnp
        z = numpy.asarray(z)
        f = numpy.sum(z ** self.degree)
        return f if patch is None else 1 + f  # all patches symmetric
