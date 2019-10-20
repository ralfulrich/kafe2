from collections import OrderedDict
from copy import deepcopy

import sys
import six

from ...core import NexusFitter, Nexus
from ...core.fitters.nexus import Parameter, Alias
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, CostFunction
from .container import UnbinnedContainer
from .cost import get_from_string
from .model import UnbinnedModelPDF, UnbinnedParametricModel
from .plot import UnbinnedPlotAdapter
from ..util import function_library, add_in_quadrature, collect, invert_matrix

__all__ = ["UnbinnedFit"]


class UnbinnedFitException(FitException):
    pass


class UnbinnedFit(FitBase):
    CONTAINER_TYPE = UnbinnedContainer
    MODEL_TYPE = UnbinnedParametricModel
    MODEL_FUNCTION_TYPE = UnbinnedModelPDF
    PLOT_ADAPTER_TYPE = UnbinnedPlotAdapter
    EXCEPTION_TYPE = UnbinnedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost', 'parameter_values', 'parameter_constraints'}


    def __init__(self,
                 data,
                 model_density_function='gaussian',
                 cost_function='nll',
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit to a model of *unbinned* data.

        :param data: the data points
        :param model_density_function: the model density
        :type model_density_function: :py:class:`~kafe2.fit.unbinned.UnbinnedModelPDF` or unwrapped native Python function
        :param cost_function: the cost function
        :param minimizer: the minimizer to use
        :param minimizer_kwargs:
        """
        FitBase.__init__(self)

        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunction):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            self._cost_function = get_from_string(cost_function)
            if self._cost_function is None:
                raise self.__class__.EXCEPTION_TYPE(
                    "Unknown cost function: %s" % cost_function)
        elif callable(cost_function):
            self._cost_function = CostFunction(cost_function)
        else:
            raise self.__class__.EXCEPTION_TYPE(
                "Invalid cost function: %s" % cost_function)

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        # retrieve fit parameter information
        self._init_fit_parameters()

        self.data = data

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)

    # private methods

    def _init_nexus(self):
        FitBase._init_nexus(self)

        # add 'x' as an alias of 'data'
        self._nexus.add_alias('x', alias_for='data')

    # -- private methods

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise UnbinnedFitException("Incompatible container type '%s' (expected '%s')"
                                       % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)

        if self._nexus:
            self._nexus.get('_external').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            data=self.data,
            model_density_function=self._model_function,
            model_parameters=self.parameter_values
        )

        if self._nexus:
            self._nexus.get('_external').mark_for_update()

    @FitBase.data.getter
    def data(self):
        """The current data of the fit object"""
        return self._data_container.data

    @property
    def data_range(self):
        """The minimum and maximum value of the data"""
        return self._data_container.data_range

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
        self._param_model.support = self.data
        return self._param_model.eval_model_function(support=x, model_parameters=model_parameters)

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super(UnbinnedFit, self).report(output_stream=output_stream,
                                        asymmetric_parameter_errors=asymmetric_parameter_errors)
