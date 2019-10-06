import numpy as np
import six

from ...core.error import MatrixGaussianError, SimpleGaussianError
from .._base import DataContainerException, DataContainerBase


__all__ = ["IndexedContainer"]


class IndexedContainerException(DataContainerException):
    pass


class IndexedContainer(DataContainerBase):
    """
    This object is a specialized data container for series of indexed measurements.

    """
    def __init__(self, data, dtype=float):
        """
        Construct a container for indexed data:

        :param data: a one-dimensional array of measurements
        :type data: iterable of type <dtype>
        :param dtype: data type of the measurements
        :type dtype: type
        """
        self._idx_data = np.array(data, dtype=dtype)
        self._error_dicts = {}
        self._total_error = None
        self._full_cor_split_system = None

    # -- private methods

    def _calculate_total_error(self):

        _sz = self.size
        _tmp_cov_mat = np.zeros((_sz, _sz))

        for _err_dict in self._error_dicts.values():
            if not _err_dict['enabled']:
                continue

            _tmp_cov_mat += _err_dict['err'].cov_mat

        self._total_error = MatrixGaussianError(
            _tmp_cov_mat, 'cov', relative=False, reference=self.data)

    def _clear_total_error_cache(self):
        self._total_error = None
        self._full_cor_split_system = None

    def _calculate_uncor_cov_mat(self):
        """calculate uncorrelated part of the covariance matrix"""

        _sz = self.size
        _tmp_uncor_cov_mat = np.zeros((_sz, _sz))

        for _err_dict in self._error_dicts.values():
            # skip disabled errors
            if not _err_dict['enabled']:
                continue

            # retrieve error object
            _err = _err_dict["err"]

            if isinstance(_err, MatrixGaussianError) or not _err_dict['splittable']:
                # cannot decorrelate full matrix errors: count as uncorrelated
                _tmp_uncor_cov_mat += _err.cov_mat
            else:
                # sum up uncorrelated parts
                _tmp_uncor_cov_mat += _err.cov_mat_uncor

        return np.array(_tmp_uncor_cov_mat)

    def _calculate_cor_nuisance_des_mat(self):
        """calculate the design matrix describing a linear map between
        the nuisance parameters for the correlated uncertainties
        and the model predictions"""

        # retrieve all fully correlated errors
        _cor_errors = self.get_matching_errors(
            matching_criteria=dict(
                enabled=True,
                correlated=True,
                splittable=True
            )
        )

        _data_size = self.size
        _err_size = len(_cor_errors)

        _des_mat = np.zeros((_err_size, _data_size))
        for _col, (_err_name, _err) in enumerate(six.iteritems(_cor_errors)):
            _des_mat[_col, :] = _err.error_cor

        return _des_mat

    def _calculate_full_cor_split_system(self):
        self._full_cor_split_system = (
            self._calculate_cor_nuisance_des_mat(),
            self._calculate_uncor_cov_mat()
        )

    # -- public properties

    @property
    def size(self):
        """number of data points"""
        return len(self._idx_data)

    @property
    def data(self):
        """container data (one-dimensional :py:obj:`numpy.ndarray`)"""
        return self._idx_data.copy()  # copy to ensure no modification by user

    @data.setter
    def data(self, data):
        _data = np.squeeze(np.array(data, dtype=float))
        if len(_data.shape) > 1:
            raise IndexedContainerException("IndexedContainer data must be 1-d array of floats! Got shape: %r..." % (_data.shape,))
        self._idx_data[:] = _data
        # reset member error references to the new data values
        for _err_dict in self._error_dicts.values():
            _err_dict['err'].reference = self._idx_data
        self._clear_total_error_cache()

    @property
    def err(self):
        """absolute total data uncertainties (one-dimensional :py:obj:`numpy.ndarray`)"""
        _total_error = self.get_total_error()
        return _total_error.error

    @property
    def cov_mat(self):
        """absolute data covariance matrix (:py:obj:`numpy.matrix`)"""
        _total_error = self.get_total_error()
        return _total_error.cov_mat

    @property
    def cov_mat_inverse(self):
        """inverse of absolute data covariance matrix (:py:obj:`numpy.matrix`), or ``None`` if singular"""
        _total_error = self.get_total_error()
        return _total_error.cov_mat_inverse

    @property
    def cor_mat(self):
        """absolute data correlation matrix (:py:obj:`numpy.matrix`)"""
        _total_error = self.get_total_error()
        return _total_error.cor_mat

    @property
    def data_range(self):
        """
        :return: the minimum and maximum value of the data
        """
        return np.amin(self.data), np.amax(self.data)


    # -- public methods

    def add_simple_error(self, err_val,
                         name=None, correlation=0, relative=False, splittable=True):
        """
        Add a simple uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param splittable: if ``False``, the error will be marked as not splittable (see `set_error_splittable`)
        :type splittable: bool or ``None``
        :return: error name
        :rtype: str
        """
        return super(IndexedContainer, self).add_simple_error(
            err_val=err_val,
            name=name,
            correlation=correlation,
            splittable=splittable,
            relative=relative,
            reference=self._idx_data  # set the reference appropriately
        )

    def add_matrix_error(self, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False):
        """
        Add a matrix uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted
                         as a *relative* uncertainty
        :type relative: bool
        :return: error name
        :rtype: str
        """
        return super(IndexedContainer, self).add_matrix_error(
            err_matrix=err_matrix,
            matrix_type=matrix_type,
            name=name,
            err_val=err_val,
            relative=relative,
            reference=self._idx_data  # set the reference appropriately
        )

