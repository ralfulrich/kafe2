import numpy as np

from . import ParametricModelBaseMixin
from .indexed import IndexedContainer, IndexedContainerException
from ...core.error import SimpleGaussianError, MatrixGaussianError

from scipy.misc import derivative

class XYContainerException(IndexedContainerException):
    pass


class XYContainer(IndexedContainer):
    AXIS_SPEC_DICT = {0:0, 1:1, '0':0, '1':1, 'x':0, 'y':1}

    def __init__(self, x_data, y_data, dtype=float):
        # TODO: check user input (?)
        self._xy_data = np.array([x_data, y_data], dtype=dtype)
        self._error_dicts = {}
        self._xy_total_errors = None


    # -- private methods

    @staticmethod
    def _find_axis_raise(axis_spec):
        try:
            axis_spec = axis_spec.lower()
        except AttributeError:
            # integers have no .lower() method
            pass
        _axis_id = XYContainer.AXIS_SPEC_DICT.get(axis_spec, None)
        if _axis_id is None:
            raise XYContainerException("No axis with id %r!" % (axis_spec,))
        return _axis_id

    def _get_data_for_axis(self, axis_id):
        return self._xy_data[axis_id]

    def _calculate_total_error(self):
        _sz = self.size
        _tmp_cov_mat_x = np.zeros((_sz, _sz))
        _tmp_cov_mat_y = np.zeros((_sz, _sz))
        for _err_dict in self._error_dicts.values():
            if not _err_dict['enabled']:
                continue
            assert _err_dict['axis'] in (0, 1)
            if _err_dict['axis'] == 0:
                _tmp_cov_mat_x += _err_dict['err'].cov_mat
            elif _err_dict['axis'] == 1:
                _tmp_cov_mat_y += _err_dict['err'].cov_mat

        _total_err_x = MatrixGaussianError(_tmp_cov_mat_x, 'cov', relative=False, reference=self.x)
        _total_err_y = MatrixGaussianError(_tmp_cov_mat_y, 'cov', relative=False, reference=self.y)
        self._xy_total_errors = [_total_err_x, _total_err_y]


    # -- public properties

    @property
    def size(self):
        return self._xy_data.shape[1]

    @property
    def data(self):
        return self._xy_data.copy()  # copy to ensure no modification by user

    @data.setter
    def data(self, new_data):
        _new_data = np.asarray(new_data)
        if _new_data.ndim != 2:
            raise XYContainerException("XYContainer data must be 2-d array of floats! Got shape: %r..." % (_new_data.shape,))
        if _new_data.shape[0] == 2:
            self._xy_data = _new_data.copy()
        elif _new_data.shape[1] == 2:
            self._xy_data = _new_data.T.copy()
        else:
            raise XYContainerException(
                "XYContainer data length must be 2 in at least one axis! Got shape: %r..." % (_new_data.shape,))

    @property
    def x(self):
        return self._get_data_for_axis(0)

    @x.setter
    def x(self, new_x):
        _new_x_data = np.squeeze(np.array(new_x))
        if len(_new_x_data.shape) > 1:
            raise XYContainerException("XYContainer 'x' data must be 1-d array of floats! Got shape: %r..." % (_new_x_data.shape,))
        self._xy_data[0,:] = new_x
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 0:
                _err_dict['err'].reference = self._get_data_for_axis(0)
        self._total_error = None

    @property
    def x_err(self):
        _total_error_x = self.get_total_error(axis=0)
        return _total_error_x.error

    @property
    def y(self):
        return self._get_data_for_axis(1)

    @y.setter
    def y(self, new_y):
        _new_y_data = np.squeeze(np.array(new_y))
        if len(_new_y_data.shape) > 1:
            raise XYContainerException("XYContainer 'y' data must be 1-d array of floats! Got shape: %r..." % (_new_y_data.shape,))
        self._xy_data[1,:] = new_y
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 1:
                _err_dict['err'].reference = self._get_data_for_axis(1)
        self._total_error = None

    @property
    def y_err(self):
        _total_error_y = self.get_total_error(axis=1)
        return _total_error_y.error

    @property
    def x_range(self):
        _x = self.x
        return np.min(_x), np.max(_x)

    @property
    def y_range(self):
        _y = self.y
        return np.min(_y), np.max(_y)


    # -- public methods

    def add_simple_error(self, axis, err_val, correlation=0, relative=False):
        _axis = self._find_axis_raise(axis)
        try:
            err_val.ndim
        except AttributeError:
            err_val = np.ones(self.size) * err_val
        _err = SimpleGaussianError(err_val=err_val, corr_coeff=correlation,
                                   relative=relative, reference=self._get_data_for_axis(_axis))
        # TODO: reason not to use id() here?
        _id = id(_err)
        assert _id not in self._error_dicts
        _new_err_dict = dict(err=_err, axis=_axis, enabled=True)
        self._error_dicts[_id] = _new_err_dict
        self._total_error = None
        return _id

    def add_matrix_error(self, axis, err_matrix, matrix_type, err_val=None, relative=False):
        _axis = self._find_axis_raise(axis)
        _err = MatrixGaussianError(err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val,
                                   relative=relative, reference=self._get_data_for_axis(_axis))
        # TODO: reason not to use id() here?
        _id = id(_err)
        assert _id not in self._error_dicts
        _new_err_dict = dict(err=_err, axis=_axis, enabled=True)
        self._error_dicts[_id] = _new_err_dict
        self._total_error = None
        return _id

    def disable_error(self, err_id):
        _err_dict = self._error_dicts.get(err_id, None)
        if _err_dict is None:
            raise XYContainerException("No error with id %d!" % (err_id,))
        _err_dict['enabled'] = False
        self._total_error = None

    def get_total_error(self, axis):
        _axis = self._find_axis_raise(axis)
        if self._xy_total_errors is None:
            self._calculate_total_error()
        return self._xy_total_errors[_axis]


class XYParametricModelException(XYContainerException):
    pass


class XYParametricModel(ParametricModelBaseMixin, XYContainer):
    def __init__(self, x_data, model_func, model_parameters):
        # print "XYParametricModel.__init__(x_data=%r, model_func=%r, model_parameters=%r)" % (x_data, model_func, model_parameters)
        _y_data = model_func(x_data, *model_parameters)
        super(XYParametricModel, self).__init__(model_func, model_parameters, x_data, _y_data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'y'
        XYContainer.y.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise XYParametricModelException("Parametric model data cannot be set!")

    @property
    def x(self):
        return super(XYParametricModel, self).x

    @x.setter
    def x(self, new_x):
        # resetting 'x' -> must reset entire data array
        self._xy_data = np.zeros((2, len(new_x)))
        self._xy_data[0] = new_x
        self._pm_calculation_stale = True

    @property
    def y(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).y

    @y.setter
    def y(self, new_y):
        raise XYParametricModelException("Parametric model data cannot be set!")

    # -- public methods

    def eval_model_function(self, x=None, model_parameters=None):
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_handle(_x, *_pars)

    def eval_model_function_derivative_by_parameters(self, x=None, model_parameters=None, par_dx=None):
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0/(1.0+np.abs(_pars)))

        _ret = []
        for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
            def _chipped_func(par):
                _chipped_pars = _pars.copy()
                _chipped_pars[_par_idx] = par
                return self._model_function_handle(_x, *_chipped_pars)

            _der_val = derivative(_chipped_func, _par_val, dx=_par_dx)
            _ret.append(_der_val)
        return np.array(_ret)

    def eval_model_function_derivative_by_x(self, x=None, model_parameters=None, dx=None):
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _dxs = dx if dx is not None else 1e-2 * (np.abs(_x) + 1.0/(1.0+np.abs(_x)))

        _ret = []
        for _x_idx, (_x_val, _dx) in enumerate(zip(_x, _dxs)):
            def _chipped_func(x):
                return self._model_function_handle(x, *_pars)

            _der_val = derivative(_chipped_func, _x_val, dx=_dx)
            _ret.append(_der_val)
        return np.array(_ret)