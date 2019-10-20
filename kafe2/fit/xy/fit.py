from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
import sys
import textwrap

from ...tools import print_dict_as_table
from ...core.fitters.nexus import Parameter, Alias, Array
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, CostFunction
from .._base.format import ModelParameterFormatter
from .container import XYContainer
from .cost import get_from_string
from .model import XYParametricModel, XYModelFunction
from .plot import XYPlotAdapter
from ..util import function_library, add_in_quadrature, string_join_if, invert_matrix, maybe_invert_matrix


__all__ = ["XYFit"]


class XYFitException(FitException):
    pass


class XYFit(FitBase):
    CONTAINER_TYPE = XYContainer
    MODEL_TYPE = XYParametricModel
    MODEL_FUNCTION_TYPE = XYModelFunction
    PLOT_ADAPTER_TYPE = XYPlotAdapter
    EXCEPTION_TYPE = XYFitException
    COST_FUNCTION_GETTER = get_from_string
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                           'x_error', 'y_data_error', 'y_model_error', 'total_error',
                           'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                           'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                           'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'
                           'y_data_uncor_cov_mat', 'y_model_uncor_cov_mat','y_total_uncor_cov_mat',
                           'nuisance_y_data_cor_cov_mat','nuisance_y_model_cor_cov_mat','nuisance_y_total_cor_cov_mat',
                           'nuisance_para', 'y_nuisance_vector',
                           'x_data_cov_mat'}

    X_ERROR_ALGORITHMS = ('iterative linear', 'nonlinear')

    AXES = ('x', 'y')

    def __init__(self,
                 xy_data,
                 model_function=function_library.linear_model,
                 cost_function='chi2',
                 x_error_algorithm='nonlinear',
                 minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit of a model to *xy* data.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.xy.XYModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunction`-derived or unwrapped native Python function
        :param x_error_algorithm: algorithm for handling x errors. Can be one of: ``'iterative linear'``, ``'nonlinear'``
        :type x_error_algorithm: str
        """

        # validate x error algorithm
        if x_error_algorithm not in XYFit.X_ERROR_ALGORITHMS:
            raise ValueError(
                "Unknown value for 'x_error_algorithm': "
                "{}. Expected one of:".format(
                    x_error_algorithm,
                    ', '.join(XYFit.X_ERROR_ALGORITHMS)
                )
            )
        else:
            self._x_error_algorithm = x_error_algorithm

        FitBase.__init__(
            self,
            model_function_spec=model_function,
            cost_function_spec=cost_function,
            minimizer=minimizer,
            minimizer_kwargs=minimizer_kwargs
        )

        # set the labels
        self.labels = [None, None]

        # set the data after the parameters, model and cost functions have been set
        self.data = xy_data


    # -- private methods

    def _init_fit_parameters(self):
        FitBase._init_fit_parameters(self)

        # -- set up nuisance parameters representing model `x` shifts

        # cannot determine x-nuisance parameters if no nexus is present
        if not self._nexus:
            return

        # only add x-nuisance parameters for the correct algorithm
        if self._x_error_algorithm != 'nuisance':
            return

        # # only fit uncorrelated x nuisance parameters if there are
        # # uncorrelated x errors (i.e. if uncor x cov mat is invertible)
        # if self._nexus.get('x_total_uncor_cov_mat_inverse').value is None:
        #     return

        # determine name of nuisance parameter array in nexus
        _nuis_vec_name = 'x_model_uncor_shift_nuis'

        # retrieve nuisance parameter array node from nexus
        _nuis_vec_node = self._nexus.get(_nuis_vec_name)

        # node not found -> no x-nuisance treatment requested skip
        if _nuis_vec_node is None:
            return

        # skip adding nuisance parameters if cost function does not depend on them
        if not _nuis_vec_node.is_descendant_of(self._nexus.get('cost')):
            return

        # append names of nuisance parameters to list of fit parameters
        _new_nuis_par_names = [
            _nuis_node.name
            for _nuis_node in _nuis_vec_node.iter_children()
        ]
        self._fit_param_names += _new_nuis_par_names

        # add a formatter for each nuisance parameter created
        _existing_formatters = self._nuisance_parameter_formatters or []
        self._nuisance_parameter_formatters = _existing_formatters + [
            ModelParameterFormatter(_nuis_par_name, tags={'nuisance'})
            for _nuis_par_name in _new_nuis_par_names
        ]

        self._fitter = None  # need to reinitialize fitter with new fit parameters

    def _init_nexus(self, nexus=None):

        nexus = FitBase._init_nexus(self, nexus)

        # -- the `y_model` implicitly depends on the `x_model` and the `poi_values`
        nexus.add_dependency(
            'y_model',
            depends_on=(
                'x_model',
                'poi_values'
            )
        )

        # -- "projected" (i.e. "x" + "y") error-related nodes

        self._add_property_to_nexus(
            'projected_xy_total_cov_mat', nexus=nexus)

        nexus.add_function(
            maybe_invert_matrix,
            func_name='projected_xy_total_cov_mat_inverse',
            par_names=('projected_xy_total_cov_mat',),
            existing_behavior='replace_if_empty'
        )

        nexus.add_dependency(
            'projected_xy_total_cov_mat',
            depends_on=(
                'poi_values',
                'x_model',
                'x_total_uncor_cov_mat',
                'y_total_uncor_cov_mat'
            )
        )

        # -- nuisance parameters due to x shifts

        _nuis_vec_name = 'x_model_uncor_shift_nuis'
        _nuis = nexus.add(
            Array([
                Parameter(0.0, name=string_join_if((_nuis_vec_name, str(i))))
                for i in range(self._data_container.size)
            ], name=_nuis_vec_name)
        )

        # `x_model` is `x_data` plus shift due to nuisance parameters
        nexus.add_function(
            #lambda data, shift: data + shift,
            lambda data, shift, ushift: data + shift + ushift,
            func_name='x_model',
            par_names=('x_data', 'x_total_cor_shift', _nuis_vec_name),
            existing_behavior='replace'
        )

        # update parametric model when `x_model` changes
        nexus.get('x_model').register_callback(
            lambda: self.MODEL_TYPE.x.fset(
                self._param_model,
                (
                    nexus.get('x_model').value
                    # + nexus.get('x_total_cor_shift').value
                    # + nexus.get('x_model_uncor_shift_nuis').value
                )
            )
        )

        # -- maybe add nuisance parameters to fit parameters

        if self._x_error_algorithm == 'nuisance':

            # add nuisance parameters to 'parameter_values' dependencies
            nexus.add_dependency(
                'parameter_values',
                depends_on=_nuis_vec_name,
            )

            # # -- register gaussian penalty term for nuisance parameters
            #
            # # get previously added nuisance parameters
            # _nuis_par_nodes = [
            #     _node.name for _node in self._nexus.get('nuisance_penalty').iter_children()
            # ]
            #
            # # add x nuisance parameters to penalty
            # self._nexus.add_function(
            #     lambda *args: np.sum([np.sum(arg**2) for arg in args]),
            #     func_name='nuisance_penalty',
            #     par_names=_nuis_par_nodes+[_nuis_vec_name],
            #     existing_behavior='replace'
            # )

        # -- re-wire nominal `y` covariance matrix (used in standard cost function)

        if self._x_error_algorithm == 'nuisance':
            nexus.add_alias(
                'y_cov_mat',
                alias_for='y_total_uncor_cov_mat',
                existing_behavior='replace'
            )
        else:
            nexus.add_alias(
                'y_cov_mat',
                alias_for='projected_xy_total_cov_mat',
                existing_behavior='replace'
            )

        # add the original function name as an alias for 'y_model'
        _func_node = nexus.get(self._model_function.name)
        if _func_node is None:
            nexus.add_alias(
                self._model_function.name,
                alias_for='y_model'
            )
        elif isinstance(_func_node, Alias):
            # remap the alias to point to 'y_model'
            _func_node.ref = nexus.get('y_model')
        elif _func_node.name == 'y_model':
            # function node exists and is called 'y_model'
            pass
        else:
            # should be prevented by correct validation
            raise AssertionError(
                "Uncaught attempt to use reserved node "
                "name '{}'!".format(_func_node.name))

        # in case 'x' errors are defined and the corresponding
        # algorithm is 'iterative linear', matrices should be projected
        # once and the corresponding node made frozen
        if self._x_error_algorithm == 'iterative linear':
            self._with_projected_nodes('freeze')

        return nexus

    def _with_projected_nodes(self, actions):
        '''perform actions on projected error nodes: freeze, update, unfreeze...'''
        if isinstance(actions, str):
            actions = (actions,)
        for _node_name in ('projected_xy_total_cov_mat',):
            for action in actions:
                _node = self._nexus.get(_node_name)
                getattr(_node, action)()

    def _calculate_y_error_band(self):
        _xmin, _xmax = self._data_container.x_range
        _band_x = np.linspace(_xmin, _xmax, 100)  # TODO: config
        _f_deriv_by_params = self._param_model.eval_model_function_derivative_by_parameters(
            x=_band_x,
            model_parameters=self.poi_values
        )
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(_band_x)
        _n_poi = len(self.poi_values)
        for _x_idx, _x_val in enumerate(_band_x):
            _p_res = _f_deriv_by_params[_x_idx]
            _band_y[_x_idx] = _p_res.dot(self.parameter_cov_mat[:_n_poi, :_n_poi]).dot(_p_res)

        return np.sqrt(_band_y)

    def _get_poi_index_by_name(self, name):
        try:
            return self._poi_names.index(name)
        except ValueError:
            raise self.EXCEPTION_TYPE('Unknown parameter name: %s' % name)

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise self.__class__.EXCEPTION_TYPE(
                "Incompatible container type '%s' (expected '%s')"
                % (type(new_data), self.CONTAINER_TYPE))
        else:
            _x_data = new_data[0]
            _y_data = new_data[1]
            self._data_container = self._new_data_container(_x_data, _y_data, dtype=float)

        self._nexus = None

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            self.x_data,
            self._model_function,
            self.poi_values
        )

        self._nexus = None

    # -- public properties

    @property
    def has_x_errors(self):
        """``True`` if at least one *x* uncertainty source has been defined"""
        return self._data_container.has_x_errors or self._param_model.has_x_errors

    @property
    def has_y_errors(self):
        """``True`` if at least one *y* uncertainty source has been defined"""
        return self._data_container.has_y_errors or self._param_model.has_y_errors

    @property
    def x_data(self):
        """array of measurement *x* values"""
        return self._data_container.x

    @property
    def x_label(self):
        """x-label to be passed on to the plot"""
        return self.labels[0]

    @x_label.setter
    def x_label(self, x_label):
        """sets the x-label to be passed onto the plot

        :param x_label: str
        """
        self.labels[0] = x_label

    @property
    def x_model(self):
        return self._param_model.x if self._param_model else self.x_data
        # if cost function uses x-nuisance parameters, consider these
        # if self._cost_function.get_flag("need_x_nuisance") and self._data_container.has_uncor_x_errors:
        #     return self.x_data + (self.x_uncor_nuisance_values * self.x_data_error)
        # else:
        #     return self.x_data

    @property
    def x_error(self):
        """array of pointwise *x* uncertainties"""
        return self._data_container.x_err

    @property
    def x_cov_mat(self):
        """the *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data(self):
        """array of measurement data *y* values"""
        return self._data_container.y

    @property
    def y_label(self):
        """y-label to be passed onto the plot"""
        return self.labels[1]

    @y_label.setter
    def y_label(self, y_label):
        """sets the y-label to be passed onto the plot

        :param y_label: str
        :return:
        """
        self.labels[1] = y_label

    @FitBase.data.getter
    def data(self):
        """(2, N)-array containing *x* and *y* measurement values"""
        return self._data_container.data

    @property
    def model(self):
        """(2, N)-array containing *x* and *y* model values"""
        return self._param_model.data

    @property
    def x_data_error(self):
        """array of pointwise *x* data uncertainties"""
        return self._data_container.x_err

    @property
    def y_data_error(self):
        """array of pointwise *y* data uncertainties"""
        return self._data_container.y_err

    @property
    def x_data_cov_mat(self):
        """the data *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data_cov_mat(self):
        """the data *y* covariance matrix"""
        return self._data_container.y_cov_mat

    @property
    def x_data_cov_mat_inverse(self):
        """inverse of the data *x* covariance matrix (or ``None`` if singular)"""
        return self._data_container.x_cov_mat_inverse

    @property
    def y_data_cov_mat_inverse(self):
        """inverse of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_cov_mat_inverse

    @property
    def x_data_cor_mat(self):
        """the data *x* correlation matrix"""
        return self._data_container.x_cor_mat

    @property
    def y_data_uncor_cov_mat(self):
        """uncorrelated part of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_uncor_cov_mat

    @property
    def y_data_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_uncor_cov_mat_inverse

    @property
    def _y_data_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all data uncertainties for all data points"""
        return self._data_container._y_nuisance_cor_design_mat

    @property
    def x_data_uncor_cov_mat(self):
        # data x uncorrelated covariance matrix
        return self._data_container.x_uncor_cov_mat

    @property
    def x_data_uncor_cov_mat_inverse(self):
        # data x uncorrelated inverse covariance matrix
        return self._data_container.x_uncor_cov_mat_inverse

    # TODO: correlated x-errors
    # @property
    # def _x_data_nuisance_cor_design_mat(self):
    #     # date x correlated matrix (nuisance)
    #     return self._data_container.nuisance_y_cor_cov_mat

    @property
    def y_data_cor_mat(self):
        """the data *y* correlation matrix"""
        return self._data_container.y_cor_mat

    @property
    def y_model(self):
        """array of *y* model predictions for the data points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y

    @property
    def x_model_error(self):
        """array of pointwise model *x* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_err

    @property
    def y_model_error(self):
        """array of pointwise model *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_err

    @property
    def x_model_cov_mat(self):
        """the model *x* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat

    @property
    def y_model_cov_mat(self):
        """the model *y* covariance matrix"""
        self._param_model.parameters = self.poi_values # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat

    @property
    def x_model_cov_mat_inverse(self):
        """inverse of the model *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat_inverse

    @property
    def y_model_cov_mat_inverse(self):
        """inverse of the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat_inverse

    @property
    def y_model_uncor_cov_mat(self):
        """uncorrelated part the model *y* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_uncor_cov_mat

    @property
    def y_model_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_uncor_cov_mat_inverse

    @property
    def x_model_uncor_cov_mat(self):
        """the model *x* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_uncor_cov_mat

    @property
    def x_model_uncor_cov_mat_inverse(self):
        """inverse of the model *x*  uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_uncor_cov_mat_inverse

    @property
    def _y_model_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all data points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model._y_nuisance_cor_design_mat

    @property
    def x_model_cor_mat(self):
        """the model *x* correlation matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    # @property TODO: correlated x-errors
    # def _x_model_nuisance_cor_design_mat(self):
    #     """model *x*  correlated covariance matrix (nuisance) (or ``None`` if singular)"""
    #     self._param_model.parameters = self.poi_values  # this is lazy, so just do it
    #     self._param_model.x = self.x_with_errors
    #     return self._param_model.nuisance_x_cor_cov_mat

    @property
    def y_model_cor_mat(self):
        """the model *y* correlation matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    @property
    def x_total_error(self):
        """array of pointwise total *x* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return add_in_quadrature(self.x_model_error, self.x_data_error)

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return add_in_quadrature(self.y_model_error, self.y_data_error)

    @property
    def projected_xy_total_error(self):
        """array of pointwise total *y* with the x uncertainties projected on top of them"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_error

        _x_errors = self.x_total_error
        _precision = 0.01 * np.min(_x_errors)
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            dx=_precision,
            model_parameters=self.poi_values
        )

        return np.sqrt(self.y_total_error**2 + self.x_total_error**2 * _derivatives**2)

    @property
    def x_total_cov_mat(self):
        """the total *x* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.x_data_cov_mat + self.x_model_cov_mat

    @property
    def y_total_cov_mat(self):
        """the total *y* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.y_data_cov_mat + self.y_model_cov_mat


    @property
    def projected_xy_total_cov_mat(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_uncor_cov_mat

        _x_errors = self.x_total_error
        _precision = 0.01 * np.min(_x_errors)
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            dx=_precision,
            model_parameters=self.poi_values
        )
        _outer_product = np.outer(_derivatives, _derivatives)

        return self.y_total_uncor_cov_mat + self.x_total_uncor_cov_mat * _outer_product


    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.x_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.y_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def projected_xy_total_cov_mat_inverse(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.projected_xy_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def y_total_uncor_cov_mat(self):
        """the total *y* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.y_data_uncor_cov_mat + self.y_model_uncor_cov_mat


    @property
    def y_total_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.y_total_uncor_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def _y_total_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all total points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self._y_data_nuisance_cor_design_mat

    @property
    def x_total_uncor_cov_mat(self):
        """the total *x* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.x_data_uncor_cov_mat + self.x_model_uncor_cov_mat

    @property
    def x_total_uncor_cov_mat_inverse(self):
        """inverse of the total *x* uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.x_total_uncor_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def y_error_band(self):
        """one-dimensional array representing the uncertainty band around the model function"""
        if not self.did_fit:
            raise XYFitException('Cannot calculate an error band without first performing a fit.')
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self._calculate_y_error_band()

    @property
    def x_range(self):
        """range of the *x* measurement data"""
        return self._data_container.x_range

    @property
    def y_range(self):
        """range of the *y* measurement data"""
        return self._data_container.y_range

    @property
    def x_uncor_nuisance_values(self):
        """gives the x uncorrelated nuisance vector"""
        _values = []
        for _name in self._x_uncor_nuisance_names:
            _values.append(self.parameter_name_value_dict[_name])
        return np.asarray(_values)

    # -- public methods

    def add_simple_error(self, axis, err_val,
                         name=None, correlation=0, relative=False, reference='data'):
        """
        Add a simple uncertainty source for axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: int
        """
        _ret = super(XYFit, self).add_simple_error(err_val=err_val,
                                                   name=name,
                                                   correlation=correlation,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # reset Nexus and Fitter (errors may be relevant for nuisance parameters)
        self._nexus = None
        self._fitter = None

        return _ret

    def add_matrix_error(self, axis, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False, reference='data'):
        """
        Add a matrix uncertainty source for an axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: int
        """
        _ret = super(XYFit, self).add_matrix_error(err_matrix=err_matrix,
                                                   matrix_type=matrix_type,
                                                   name=name,
                                                   err_val=err_val,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # reset Nexus and Fitter (errors may be relevant for nuisance parameters)
        self._nexus = None
        self._fitter = None

        return _ret

    def set_poi_values(self, param_values):
        """set the start values of all parameters of interests"""
        _param_names = self._poi_names
        #test list length
        if not len(param_values) == len(_param_names):
            raise XYFitException("Cannot set all fit parameter values: %d fit parameters declared, "
                                       "but %d provided!"
                                       % (len(_param_names), len(param_values)))
        # set values in nexus
        _par_val_dict = {_pn: _pv for _pn, _pv in zip(_param_names, param_values)}
        self.set_parameter_values(**_par_val_dict)

    def do_fit(self):
        """Perform the fit."""
        if self._cost_function.needs_errors and not self._data_container.has_y_errors:
            self._cost_function.on_no_errors()

        # initialize the Nexus (if not done already)
        if not self._nexus:
            self._init_nexus()

        self._init_fit_parameters()  # ensure parameters are up to date

        # initialize the fitter (if not done already)
        if not self._fitter:
            self._initialize_fitter(self._minimizer, self._minimizer_kwargs)

        # explicitly update (frozen) projected covariance matrix before fit
        self._with_projected_nodes('update')

        if self.has_x_errors:
            if self._x_error_algorithm == 'nonlinear':
                # 'nonlinear' x error fitting: one iteration;
                # projected covariance matrix is updated during minimization
                self._with_projected_nodes(('update', 'unfreeze'))
                super(XYFit, self).do_fit()

            elif self._x_error_algorithm == 'nuisance':
                super(XYFit, self).do_fit()

            elif self._x_error_algorithm == 'iterative linear':
                # 'iterative linear' x error fitting: multiple iterations;
                # projected covariance matrix is only updated in-between
                # and kept constant during minimization
                self._with_projected_nodes(('update', 'freeze'))

                # perform a preliminary fit
                self._fitter.do_fit()

                # iterate until cost function value converges
                _convergence_limit = float(kc('fit', 'x_error_fit_convergence_limit'))
                _previous_cost_function_value = self.cost_function_value
                for i in range(kc('fit', 'max_x_error_fit_iterations')):

                    # explicitly update (frozen) projected matrix before each iteration
                    self._with_projected_nodes('update')

                    self._fitter.do_fit()

                    # check convergence
                    if np.abs(self.cost_function_value - _previous_cost_function_value) < _convergence_limit:
                        break  # fit converged

                    _previous_cost_function_value = self.cost_function_value
            else:
                raise NotImplementedError(
                    "No implementation for x error algorithm '{}'".format(self._x_error_algorithm))
        else:
            # no 'x' errors: fit as usual

            # freeze error projection nodes (faster)
            self._with_projected_nodes(('update', 'freeze'))

            super(XYFit, self).do_fit()

        # explicitly update error projection nodes
        self._with_projected_nodes('update')

        # clear loaded results and update parameter formatters
        self._loaded_result_dict = None
        self._update_parameter_formatters()

    def eval_model_function(self, x=None, model_parameters=None):
        """
        Evaluate the model function.

        :param x: values of *x* at which to evaluate the model function (if ``None``, the data *x* values are used)
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)

    def calculate_nuisance_parameters(self):
        """
        Calculate and return the nuisance parameter values.

        NOTE: this only calculates nuisance parameters
        for correlated 'x` and `y' uncertainties.

        :return: vector containing the nuisance parameter values
        :rtype: ``numpy.array``
        """

        _shifts = []
        for _cont in (self._data_container, self._param_model):
            try:
                _shifts.extend(
                    _cont.get_shift_coefficients('y', self.y_data - self.y_model))
            except np.linalg.LinAlgError:
                # matrix is singular
                pass

        for _cont in (self._data_container, self._param_model):
            try:
                _shifts.extend(
                    _cont.get_shift_coefficients('x', self.x_data - self.x_model))
            except np.linalg.LinAlgError:
                # matrix is singular
                pass

        return _shifts

    def generate_plot(self):
        _plot = super(XYFit, self).generate_plot()
        _plot.x_label = self.x_label
        _plot.y_label = self.y_label
        return _plot

    def report(self, output_stream=sys.stdout,
               show_data=True,
               show_model=True,
               asymmetric_parameter_errors=False):
        """
        Print a summary of the fit state and/or results.

        :param output_stream: the output stream to which the report should be printed
        :type output_stream: TextIOBase
        :param show_data: if ``True``, print out information about the data
        :type show_data: bool
        :param show_model: if ``True``, print out information about the parametric model
        :type show_model: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """
        _indent = ' ' * 4

        if show_data:
            output_stream.write(textwrap.dedent("""
                ########
                # Data #
                ########

            """))
            _data_table_dict = OrderedDict()
            _data_table_dict['X Data'] = self.x_data
            if self._data_container.has_x_errors:
                _data_table_dict['X Data Error'] = self.x_data_error
                #_data_table_dict['X Data Total Covariance Matrix'] = self.x_data_cov_mat
                _data_table_dict['X Data Total Correlation Matrix'] = self.x_data_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)
            output_stream.write('\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['Y Data'] = self.y_data
            if self.has_data_errors:
                _data_table_dict['Y Data Error'] = self.y_data_error
                #_data_table_dict['Y Data Total Covariance Matrix'] = self.y_data_cov_mat
                _data_table_dict['Y Data Total Correlation Matrix'] = self.y_data_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        if show_model:
            output_stream.write(textwrap.dedent("""
                #########
                # Model #
                #########

            """))

            #output_stream.write(_indent)
            output_stream.write(_indent + "Model Function\n")
            output_stream.write(_indent + "==============\n\n")
            output_stream.write(_indent * 2)
            output_stream.write(
                self._model_function.formatter.get_formatted(
                    with_par_values=False,
                    n_significant_digits=2,
                    format_as_latex=False,
                    with_expression=True
                )
            )
            output_stream.write('\n\n\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['X Model'] = self.x_model
            if self.has_model_errors:
                _data_table_dict['X Model Error'] = self.x_model_error
                #_data_table_dict['X Model Total Covariance Matrix'] = self.x_model_cor_mat
                _data_table_dict['X Model Total Correlation Matrix'] = self.x_model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)
            output_stream.write('\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['Y Model'] = self.y_model
            if self.has_model_errors:
                _data_table_dict['Y Model Error'] = self.y_model_error
                #_data_table_dict['Y Model Total Covariance Matrix'] = self.y_model_cov_mat
                _data_table_dict['Y Model Total Correlation Matrix'] = self.y_model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        super(XYFit, self).report(output_stream=output_stream, asymmetric_parameter_errors=asymmetric_parameter_errors)
