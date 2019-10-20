from collections import OrderedDict
from copy import deepcopy

import numpy as np
import sys
import textwrap

from ...tools import print_dict_as_table
from .._base import FitException, FitBase, DataContainerBase, CostFunction
from .container import IndexedContainer
from .cost import get_from_string
from .model import IndexedParametricModel, IndexedModelFunction
from .plot import IndexedPlotAdapter
from ..util import function_library, add_in_quadrature, invert_matrix


__all__ = ["IndexedFit"]


class IndexedFitException(FitException):
    pass


class IndexedFit(FitBase):
    CONTAINER_TYPE = IndexedContainer
    MODEL_TYPE = IndexedParametricModel
    MODEL_FUNCTION_TYPE = IndexedModelFunction
    PLOT_ADAPTER_TYPE = IndexedPlotAdapter
    EXCEPTION_TYPE = IndexedFitException
    COST_FUNCTION_GETTER = get_from_string
    RESERVED_NODE_NAMES = {'data', 'model', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}


    def __init__(self,
                 data,
                 model_function,
                 cost_function='chi2',
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of a model to a series of indexed measurements.

        :param data: the measurement values
        :type data: iterable of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.indexed.IndexedModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunction`-derived or unwrapped native Python function
        """
        FitBase.__init__(
            self,
            model_function_spec=model_function,
            cost_function_spec=cost_function,
            minimizer=minimizer,
            minimizer_kwargs=minimizer_kwargs
        )

        # set the data after the parameters, model and cost functions have been set
        self.data = data

    # -- private methods

    def _init_nexus(self):
        FitBase._init_nexus(self)

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise IndexedFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)

        if self._nexus is not None:
            self._nexus.get('_external').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            self._model_function,
            self.poi_values,
            shape_like=self.data
        )

        if self._nexus is not None:
            self._nexus.get('_external').mark_for_update()

    # -- public properties

    @FitBase.data.getter
    def data(self):
        """array of measurement values"""
        return self._data_container.data

    @property
    def data_error(self):
        """array of pointwise data uncertainties"""
        return self._data_container.err

    @property
    def data_cov_mat(self):
        """the data covariance matrix"""
        return self._data_container.cov_mat

    @property
    def data_cov_mat_inverse(self):
        """inverse of the data covariance matrix (or ``None`` if singular)"""
        return self._data_container.cov_mat_inverse

    @property
    def data_cor_mat(self):
        """the data correlation matrix"""
        return self._data_container.cor_mat

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err

    @property
    def model_cov_mat(self):
        """the model covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat

    @property
    def model_cov_mat_inverse(self):
        """inverse of the model covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat_inverse

    @property
    def model_cor_mat(self):
        """the model correlation matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cor_mat

    @property
    def total_error(self):
        """array of pointwise total uncertainties"""
        return add_in_quadrature(self.data_error, self.model_error)

    @property
    def total_cov_mat(self):
        """the total covariance matrix"""
        return self.data_cov_mat + self.model_cov_mat

    @property
    def total_cov_mat_inverse(self):
        """inverse of the total covariance matrix (or ``None`` if singular)"""
        return invert_matrix(self.total_cov_mat)

    # -- public methods

    ## add_error... methods inherited from FitBase ##

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
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Data'] = self.data
            if self.has_data_errors:
                _data_table_dict['Data Error'] = self.data_error
                #_data_table_dict['Data Total Covariance Matrix'] = self.data_cov_mat
                _data_table_dict['Data Total Correlation Matrix'] = self.data_cor_mat

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
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Model'] = self.model
            if self.has_model_errors:
                _data_table_dict['Model Error'] = self.model_error
                #_data_table_dict['Model Total Covariance Matrix'] = self.model_cov_mat
                _data_table_dict['Model Total Correlation Matrix'] = self.model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        super(IndexedFit, self).report(output_stream=output_stream,
                                       asymmetric_parameter_errors=asymmetric_parameter_errors)
