import numpy as np
import six
import uuid

from .format import ModelParameterFormatter, CostFunctionFormatter

from kafe2.fit.io.file import FileIOMixin


if six.PY2:
    from funcsigs import Parameter, signature as _get_signature
else:
    from inspect import Parameter, signature as _get_signature


def remap_parameters(par_map):
    """Decorator for remapping function parameter names. Takes a function and returns
    a version with a different signature that calls the original function."""
    import re
    PAR_REGEXP = re.compile('^[A-Za-z_][A-Za-z0-9_]*$')

    def make_remapped_func(func):
        s = _get_signature(func)

        _new_sig, _old_sig = [], []
        for _old, _old_par in s.parameters.items():
            _new = par_map.get(_old, _old)
            if _new is None:
                # remove from new signature
                continue

            _new_default = _old_par.default
            if isinstance(_new, tuple):
                assert len(_new) == 2
                _new, _new_default = _new

            if isinstance(_new, list):
                for _new_elem in _new:
                    # prevent non-identifiers
                    _new_elem = _new_elem.strip()
                    if not re.match(PAR_REGEXP, _new_elem):
                        raise ValueError("Invalid parameter name: '{}'".format(_new))

                    _new_sig.append((_new_elem,))
                _old_sig.append((_old, '[{}]'.format(', '.join([_new_elem.strip() for _new_elem in _new]))))
            else:
                # prevent non-identifiers
                _new = _new.strip()
                if not re.match(PAR_REGEXP, _new):
                    raise ValueError("Invalid parameter name: '{}'".format(_new))

                if _new_default != "__empty__" and _new_default != Parameter.empty:
                    _new_sig.append((_new, _new_default))
                else:
                    _new_sig.append((_new,))
                _old_sig.append((_old, _new))

        # TODO: implement without `lambda`
        _code = "lambda {}: {}({})".format(
            ', '.join([
                "{}={}".format(_new_spec[0], _new_spec[1])
                if len(_new_spec) == 2
                else "{}".format(_new_spec[0],)
                for _new_spec in _new_sig
            ]),
            func.__name__,
            ', '.join([
                "{}={}".format(_old_spec[0], _old_spec[1])
                if len(_old_spec) == 2
                else "{}".format(_old_spec[0],)
                for _old_spec in _old_sig
            ])
        )

        return eval(_code, {func.__name__: func})

    return make_remapped_func


class CostFunctionException(Exception):
    pass


class CostFunction(FileIOMixin, object):
    """
    This is a base class implementing the minimal interface required by all
    cost functions.

    Any Python function returning a ``float`` can be used as a cost function,
    although a number of common cost functions are provided as built-ins for
    all fit types.

    In order to be used as a cost function, a native Python function must be wrapped
    by an object whose class derives from this base class.

    This class provides the basic functionality used by all :py:class:`CostFunction` objects.
    These use introspection (:py:mod:`inspect`) for determining the parameter structure of the
    cost function and to ensure the function can be used as a cost function (validation).
    """

    EXCEPTION_TYPE = CostFunctionException
    FORMATTER_TYPE = CostFunctionFormatter

    def __init__(self, cost_function):
        """
        Construct :py:class:`CostFunction` object (a wrapper for a native Python function):

        :param cost_function: function handle
        :param signature: `inspect.Signature` object representing the function signature. If
            not given, it is inferred from the function itself
        """
        self._func = cost_function

        self._signature = _get_signature(self._func)

        self._name = self._func.__name__

        # make lambda names valid Python identifiers
        if self._name == '<lambda>':
            self._name = 'lambda_' + uuid.uuid4().hex[:10]

        self._validate_cost_function_raise()
        self._assign_function_formatter()

        self._ndf = None  # number of degrees of freedom in fit
        self._needs_errors = True
        self._no_errors_warning_printed = False

        super(CostFunction, self).__init__()

    def __add__(self, other):
        '''add together cost functions'''
        # turn non-callable other into trivial callable
        if not callable(other):
            def other():
                return other

        # cast to cost function if needed
        if not isinstance(other, CostFunction):
            other = CostFunction(other)

        _common_parameters = set(self.parameter_names).intersection(set(other.parameter_names))
        if _common_parameters:
            raise ValueError(
                "Cannot add together cost functions "
                "with common parameters: {!r}".format(_common_parameters))

        @remap_parameters({'args1': list(self.parameter_names), 'args2': list(other.parameter_names)})
        def _cfunc(args1, args2):
            return self.func(*args1) + other.func(*args2)

        return CostFunction(_cfunc)

    @classmethod
    def _get_base_class(cls):
        return CostFunction

    @classmethod
    def _get_object_type_name(cls):
        return 'cost_function'

    def _validate_cost_function_raise(self):
        if 'cost' in self._signature.parameters:
            raise self.__class__.EXCEPTION_TYPE(
                "The alias 'cost' for the cost function value cannot be used as an argument to the cost function!")

        # evaluate general cost function requirements
        for _par in self._signature.parameters.values():
            if _par.kind == _par.VAR_POSITIONAL:
                raise self.__class__.EXCEPTION_TYPE(
                    "Cost function '{}' with variable number of positional "
                    "arguments (*{}) is not supported".format(
                        self._func.__name__,
                        _par.name,
                    )
                )
            elif _par.kind == _par.VAR_KEYWORD:
                raise self.__class__.EXCEPTION_TYPE(
                    "Cost function '{}' with variable number of keyword "
                    "arguments (**{}) is not supported".format(
                        self._func.__name__,
                        _par.name,
                    )
                )

    def _get_parameter_formatters(self):
        return [ModelParameterFormatter(name=_pn, value=None, error=None)
                for _pn in self.signature.parameters]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(
            self.name, arg_formatters=self._get_parameter_formatters())

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    @property
    def name(self):
        """The cost function name (a valid Python identifier)"""
        return self._name

    @property
    def func(self):
        """The cost function handle"""
        return self._func

    @property
    def signature(self):
        """The model function argument specification, as returned by :py:meth:`inspect.signature`"""
        return self._signature

    @property
    def parameter_names(self):
        return tuple(self._signature.parameters)

    @property
    def formatter(self):
        """The :py:obj:`Formatter` object for this function"""
        return self._formatter

    @property
    def argument_formatters(self):
        """The :py:obj:`Formatter` objects for the function arguments"""
        return self._formatter.arg_formatters

    @property
    def ndf(self):
        """The number of degrees of freedom of this cost function"""
        return self._ndf

    @ndf.setter
    def ndf(self, new_ndf):
        """The number of degrees of freedom of this cost function"""
        assert new_ndf > 0  # ndf must be positive
        assert new_ndf == int(new_ndf)  # ndf must be integer
        self._ndf = new_ndf

    @property
    def needs_errors(self):
        """Whether the cost function needs errors for a meaningful result"""
        return self._needs_errors

    def get_uncertainty_gaussian_approximation(self, data):
        """
        Get the gaussian approximation of the uncertainty inherent to the cost function, returns 0 by default.
        :param data: the fit data
        :return: the approximated gaussian uncertainty given the fit data
        """
        return 0

    def set_flag(self, name, value):
        self._flags[name] = value

    def get_flag(self, name):
        return self._flags.get(name, None)

    def is_data_compatible(self, data):
        """
        Tests if model data is compatible with cost function
        :param data: the fit data
        :type data: numpy.ndarray
        :return: if the data is compatible, and if not a reason for the incompatibility
        :rtype: (boo, str)
        """
        return True, None

    def on_no_errors(self):
        if not self._no_errors_warning_printed:
            print('WARNING: No data errors were specified. The fit results may be wrong.')


# -- cost function contributinos

@CostFunction
def constraints_penalty(poi_values, parameter_constraints):
    """Generic penalty term due to parameter constraints"""
    if parameter_constraints is not None:
        return np.sum([_c.cost(poi_values) for _c in parameter_constraints])
    return 0


@CostFunction
def nuisance_penalty(nuisance_penalty):
    """Generic penalty term due to nuisance parameters"""
    return nuisance_penalty


# -- cost function factories

def make_chi2_cost_function(data_name, model_name, shift_name=None,
                            cov_mat_inverse_name=None, description=None,
                            singular_behavior='raise'):
    """Cost function factory. Constructs an appropriate signature"""

    if callable(singular_behavior):
        _handle_singular = singular_behavior
    elif singular_behavior == 'raise':
        def _handle_singular(mat=None):
            raise np.linalg.LinAlgError("Singular matrix!")
    elif singular_behavior == 'ones':
        def _handle_singular(mat=None):
            return 1.0
    elif singular_behavior == 'zeros':
        def _handle_singular(mat=None):
            return 0.0
    elif singular_behavior == 'inf':
        def _handle_singular(mat=None):
            return np.inf
    else:
        raise ValueError(
            "Unknown specification '{}' for `singular_behavior`: "
            "expected a callable or one of {!r}".format(
                singular_behavior,
                {'raise', 'ones', 'zeros', 'inf'}
            )
        )

    @remap_parameters(dict(
        data=data_name,
        model=model_name,
        shift=shift_name,
        cov_mat_inverse=cov_mat_inverse_name))
    def _cost(data, model, shift=None, cov_mat_inverse=None):

        data = np.asarray(data)
        model = np.asarray(model)

        if model.shape != data.shape:
            raise CostFunctionException("'data' and 'model' must have the same shape! Got %r and %r..."
                                        % (data.shape, model.shape))

        _res = (data - model)

        if shift is not None:
            _res -= shift

        if cov_mat_inverse is None:
            cov_mat_inverse = _handle_singular()

        # if a covariance matrix inverse is given, use it
        return _res.dot(np.asarray(cov_mat_inverse)).dot(_res)

    _cf = CostFunction(_cost)

    if description:
        _cf.formatter.description = description

    return _cf


def make_nll_cost_function(data_name, model_name, log_pdf, use_likelihood_ratio=True, description=None):
    """Cost function factory. Constructs an appropriate signature"""

    @remap_parameters(dict(data=data_name, model=model_name))
    def _cost(data, model):
        data = np.asarray(data)
        model = np.asarray(model)

        if model.shape != data.shape:
            raise CostFunctionException("'data' and 'model' must have the same shape! Got %r and %r..."
                                        % (data.shape, model.shape))

        _nll = - 2.0 * np.sum(log_pdf(data, model))

        # add back the neg-log-likelihood of a perfect match (TODO)
        if use_likelihood_ratio:
            pass #_nll += 2.0 * np.sum(log_pdf(data, data))

        return _nll

    _cf = CostFunction(_cost)

    if description:
        _cf.formatter.description = description

    return _cf


def make_getter_function_from_string(name_func_map):
    def _lookup(key):
        # normalize key before lookup
        key = key.lower().replace('_', '').replace(' ', '')
        return name_func_map.get(key, None)
    return _lookup
