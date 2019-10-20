from copy import deepcopy

from .._base import FitException, FitBase, DataContainerBase, CostFunction
from ..util import function_library, add_in_quadrature, invert_matrix

from .container import HistContainer
from .cost import get_from_string
from .model import HistParametricModel, HistModelFunction
from .plot import HistPlotAdapter

__all__ = ["HistFit"]


class HistFitException(FitException):
    pass


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    MODEL_FUNCTION_TYPE = HistModelFunction
    PLOT_ADAPTER_TYPE = HistPlotAdapter
    EXCEPTION_TYPE = HistFitException
    COST_FUNCTION_GETTER = get_from_string
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}

    def __init__(self,
                 data,
                 model_density_function=function_library.normal_distribution_pdf,
                 cost_function='nll',
                 model_density_antiderivative=None,
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of a model to a histogram.

        :param data: the measurement values
        :type data: iterable of float
        :param model_density_function: the model density function
        :type model_density_function: :py:class:`~kafe2.fit.hist.HistModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunction`-derived or unwrapped native Python function
        """
        FitBase.__init__(
            self,
            model_function_spec=(model_density_function, model_density_antiderivative),
            cost_function_spec=cost_function,
            minimizer=minimizer,
            minimizer_kwargs=minimizer_kwargs
        )

        # set the data after the parameters, model and cost functions have been set
        self.data = data

    # -- private methods

    def _init_model_function(self, model_function_spec):
        model_density_function, model_density_antiderivative = model_function_spec

        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            # TODO shouldn't this Exception only be raised if the kafe2 model function already has an antiderivative?
            if model_density_antiderivative is not None:
                raise HistFitException("Antiderivative (%r) provided in constructor for %r, "
                                       "but histogram model function object (%r) already constructed!"
                                       % (model_density_antiderivative, self.__class__, model_density_function))
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(
                model_density_function,
                model_density_antiderivative=model_density_antiderivative
            )

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

    def _init_nexus(self):
        FitBase._init_nexus(self)

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise HistFitException("Incompatible container type '{}' (expected '{}')"
                                   .format(type(new_data), self.CONTAINER_TYPE))
        else:
            raise HistFitException("Fitting a histogram requires a HistContainer!")

        if self._nexus is not None:
            self._nexus.get('_external').mark_for_update()

    def _set_new_parametric_model(self):
        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(
            self._data_container.size,
            self._data_container.bin_range,
            self._model_function,
            self.poi_values,
            self._data_container.bin_edges,
            model_density_func_antiderivative=
                self._model_function.antiderivative
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
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data * self._data_container.n_entries  # NOTE: model is just a density->scale up

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err  # FIXME: how to handle scaling

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

    def eval_model_function_density(self, x, model_parameters=None):
        """
        Evaluate the model function density.

        :param x: values of *x* at which to evaluate the model function density
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function density values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.eval_model_function_density(x=x, model_parameters=model_parameters)
