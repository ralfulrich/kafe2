import numpy as np
import unittest2 as unittest

from kafe2.test.fit import KafeAssertionsMixin

from scipy import stats

from kafe2.core.constraint import GaussianSimpleParameterConstraint, GaussianMatrixParameterConstraint

from kafe2.fit._base.cost import make_chi2_cost_function, make_nll_cost_function, constraints_penalty


class TestCostFunctionBase(KafeAssertionsMixin):

    ARGDICT_TEMPLATE = dict()

    def __init__(self, *args, **kwargs):
        super(TestCostFunctionBase, self).__init__(*args, **kwargs)

    def setUp(self):
        self._tol = 1e-10

        self._cost_values = {}

        self._par_vals = np.array([12.3, 0.001, -1.9])

        self._simple_constraint = GaussianSimpleParameterConstraint(
            index=0, value=10.0, uncertainty=2.5)
        self._matrix_constraint = GaussianMatrixParameterConstraint(
            indices=(0, 1, 2),
            values=(1.0, 2.0, 3.0),
            matrix=[
                [1.5, 0.1, 0.1],
                [0.1, 2.2, 0.1],
                [0.1, 0.1, 0.3]
            ]
        )
        self._par_constraints = [self._simple_constraint, self._matrix_constraint]
        self._par_cost = self._simple_constraint.cost(self._par_vals) + self._matrix_constraint.cost(self._par_vals)

        self._test_spec = {}

    def test(self):
        for _test_name, _spec in self._test_spec.items():
            with self.subTest(name=_test_name):

                # check value without parameter constraints
                _value = None
                try:
                    _value = _spec['cost_function'](
                            **dict(self.ARGDICT_TEMPLATE,
                                   **_spec['argdict']))
                except:
                    pass  # on error, leave value as `None` for comparison

                self._assert_compatible(
                    value=_value,
                    reference=_spec['reference_value'],
                    name='cost_function',
                    atol=self._tol
                )

                # check value with parameter constraints
                _value_with_pc = None
                try:
                    _value_with_pc = _spec['cost_function'](
                            **dict(self.ARGDICT_TEMPLATE,
                                   poi_values=self._par_vals,
                                   parameter_constraints=self._par_constraints,
                                   **_spec['argdict']))
                except:
                    pass  # on error, leave value as `None` for comparison

                _ref_val = _spec['reference_value']
                if _ref_val is not None:
                    _ref_val += self._par_cost

                self._assert_compatible(
                    value=_value_with_pc,
                    reference=_ref_val,
                    name='cost_function',
                    atol=self._tol
                )


class TestCostFunctionChi2(TestCostFunctionBase, unittest.TestCase):

    ARGDICT_TEMPLATE = dict(
        data=None, model=None,
        cov_mat_inverse=None,
        poi_values=None, parameter_constraints=None
    )

    def setUp(self):
        TestCostFunctionBase.setUp(self)

        self._data = np.array([-0.5, 2.1, 8.9])
        self._model = np.array([5.7, 8.4, -2.3])

        self._cov_mat = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0]
        ])
        self._cov_mat_inv = np.linalg.inv(self._cov_mat)
        self._pointwise_errors = np.sqrt(np.diag(self._cov_mat))

        self._res = self._data - self._model

        _generic_chi2 = make_chi2_cost_function(
            data_name='data',
            model_name='model',
            cov_mat_inverse_name='cov_mat_inverse',
            singular_behavior='ones'
        ) + constraints_penalty

        self._test_spec = dict(
            no_cov_mat_inverse_ones=dict(
                cost_function=make_chi2_cost_function(
                    data_name='data',
                    model_name='model',
                    cov_mat_inverse_name='cov_mat_inverse',
                    singular_behavior='ones'
                ) + constraints_penalty,
                argdict=dict(
                    data=self._data,
                    model=self._model),
                reference_value=np.sum(self._res ** 2)
            ),
            no_cov_mat_inverse_raise=dict(
                cost_function=make_chi2_cost_function(
                    data_name='data',
                    model_name='model',
                    cov_mat_inverse_name='cov_mat_inverse',
                    singular_behavior='raise'
                ) + constraints_penalty,
                argdict=dict(
                    data=self._data,
                    model=self._model),
                reference_value=None
            ),
            cov_mat=dict(
                cost_function=make_chi2_cost_function(
                    data_name='data',
                    model_name='model',
                    cov_mat_inverse_name='cov_mat_inverse',
                    singular_behavior='raise'
                ) + constraints_penalty,
                argdict=dict(
                    data=self._data,
                    model=self._model,
                    cov_mat_inverse=self._cov_mat_inv),
                reference_value=self._res.dot(self._cov_mat_inv).dot(self._res),
            ),
        )


class TestCostFunctionNLL(TestCostFunctionBase, unittest.TestCase):

    ARGDICT_TEMPLATE = dict(
        data=None, model=None,
        poi_values=None, parameter_constraints=None
    )

    def setUp(self):
        TestCostFunctionBase.setUp(self)

        self._data = np.array([0.0, 2.0, 9.0])
        self._model = np.array([5.7, 8.4, 2.3])

        _generic_chi2 = make_chi2_cost_function(
            data_name='data',
            model_name='model',
            cov_mat_inverse_name='cov_mat_inverse'
        )

        self._test_spec = dict(
            poisson=dict(
                cost_function=make_nll_cost_function(
                    data_name='data',
                    model_name='model',
                    log_pdf=stats.poisson.logpmf
                ) + constraints_penalty,
                argdict=dict(
                    data=self._data,
                    model=self._model),
                reference_value=-2 * np.sum(stats.poisson.logpmf(self._data, self._model))
            ),
        )
