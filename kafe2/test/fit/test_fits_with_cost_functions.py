import numpy as np
import unittest2 as unittest

from parameterized import parameterized

from kafe2.fit import XYFit, XYContainer, string_join_if
from kafe2.fit.xy.cost import chi2

from kafe2.test.fit.test_fit import KafeFitPropertiesAssertionMixin


class TestXYFitWithCostChi2Full(KafeFitPropertiesAssertionMixin, unittest.TestCase):

    MINIMIZER = 'scipy'

    PARAMETER_SPEC = []

    _X_ALGOS = XYFit.X_ERROR_ALGORITHMS

    _ERR_SPECS = ('none', 'uncor', 'partcor', 'fullcor')

    for _algo in _X_ALGOS:
        for _x_err_spec in _ERR_SPECS:
            for _y_err_spec in _ERR_SPECS:
                _key = string_join_if((_algo, 'x', _x_err_spec, 'y', _y_err_spec))

                _make_fit_kwargs = dict(
                    x_error_algorithm=_algo,
                    x_error_spec=_x_err_spec,
                    y_error_spec=_y_err_spec,
                    do_fit=True
                )

                PARAMETER_SPEC.append((_key, _make_fit_kwargs))

    def _make_fit(self, x_error_algorithm=None, x_error_spec=None, y_error_spec=None, do_fit=False):
        '''construct a fit based on test parameters'''

        assert isinstance(x_error_spec, str)
        assert isinstance(y_error_spec, str)

        # -- return `None` for untestable cases

        # # 'nuisance' requires the `y` matrices to be invertible
        # if x_error_algorithm == 'nuisance':
        #     if y_error_spec in ('none', 'fullcor', 'fullnuisancecor'):
        #         return None
        # else:
        #     # other fit algorithms require at least one of `x` and `y` matrices to be invertible
        #     if (
        #         x_error_spec in ('none', 'fullcor', 'fullnuisancecor') and
        #         y_error_spec in ('none', 'fullcor', 'fullnuisancecor')
        #     ):
        #         return None

        _fit = XYFit(
            xy_data=XYContainer(
                x_data=[1.0, 2.0, 3.0, 4.0],
                y_data=[2.3, 4.2, 7.5, 9.4],
            ),
            cost_function=chi2,
            x_error_algorithm=x_error_algorithm,
            model_function=lambda x, a: a*x,
            minimizer=self.MINIMIZER
        )

        for _spec, _axis in zip((x_error_spec, y_error_spec), ('x', 'y')):
            if _spec not in self._ERR_SPECS:
                raise ValueError("Unknown error specification: {}".format(_spec))

            if 'uncor' in _spec:
                _fit.add_simple_error(_axis, 1.0)
            if 'partcor' in _spec:
                _fit.add_simple_error(_axis, 1.0, correlation=0.0)
                _fit.add_simple_error(_axis, 1.0, correlation=1.0)
            if 'fullcor' in _spec:
                _fit.add_simple_error(_axis, 1.0, correlation=1.0)
            if 'partnuisancecor' in _spec:
                raise NotImplementedError
            if 'fullnuisancecor' in _spec:
                raise NotImplementedError

        if do_fit:
            try:
                _fit.do_fit()
            except Exception as e:
                # fit failed -> keep fit object for testing
                pass

        return _fit

    def _get_ref_par_values(self, x_error_algorithm, x_error_spec, y_error_spec, do_fit):

        xe, ye = x_error_spec, y_error_spec  # convenience
        algo = x_error_algorithm  # convenience

        if do_fit:
            if xe == 'none':
                if ye == 'none' or ye == 'fullcor' or ye == 'fullnuisancecor' or ye == 'uncor':
                    if algo == 'iterative linear': return [2.353]
                    elif algo == 'nonlinear':      return [2.353]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.393]
                    elif algo == 'nonlinear':      return [2.393]
                    elif algo == 'nuisance':       raise NotImplementedError
            elif xe == 'uncor':
                if ye == 'none':
                    if algo == 'iterative linear': return [2.353]
                    elif algo == 'nonlinear':      return [2.360]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.353]
                    elif algo == 'nonlinear':      return [2.359]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.378]
                    elif algo == 'nonlinear':      return [2.370]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.393]
                    elif algo == 'nonlinear':      return [2.373]
                    elif algo == 'nuisance':       raise NotImplementedError
            elif xe == 'partcor' or xe == 'partnuisancecor':
                if ye == 'none':
                    if algo == 'iterative linear': return [2.393]
                    elif algo == 'nonlinear':      return [2.411]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.378]
                    elif algo == 'nonlinear':      return [2.404]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.393]
                    elif algo == 'nonlinear':      return [2.409]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.410]
                    elif algo == 'nonlinear':      return [2.416]
                    elif algo == 'nuisance':       raise NotImplementedError
            elif xe == 'fullcor' or xe == 'fullnuisancecor':
                if ye == 'none' or ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.353]
                    elif algo == 'nonlinear':      return [2.353]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.393]
                    elif algo == 'nonlinear':      return [2.434]
                    elif algo == 'nuisance':       raise NotImplementedError
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.410]
                    elif algo == 'nonlinear':      return [2.436]
                    elif algo == 'nuisance':       raise NotImplementedError

        return [9.999]  # fell through -> return dummy

    def _get_ref(self, fit, x_error_algorithm, x_error_spec, y_error_spec, do_fit):

        xe, ye = x_error_spec, y_error_spec  # convenience
        algo = x_error_algorithm  # convenience

        for _spec in (xe, ye):
            if _spec not in self._ERR_SPECS:
                raise ValueError("Unknown error specification: {}".format(_spec))

        _x_cor_nuis_names, _y_cor_nuis_names, _x_nuis_names = tuple(), tuple(), tuple()

        _ref_poi_values = self._get_ref_par_values(
            x_error_algorithm, x_error_spec, y_error_spec, do_fit)

        if 'nuisancecor' in xe:
            _x_cor_nuis_names= ('x_data_cor_shift_nuis_0',)

        if 'nuisancecor' in ye:
            _y_cor_nuis_names= ('y_data_cor_shift_nuis_0',)

        # x nuisance parameters are only fitted
        # if the x error algorithm is 'nuisance'
        # and the x error matrix is invertible
        if (
            algo == 'nuisance' and
            ('uncor' in xe or
             'partcor' in xe or
             'partnuisancecor' in xe)):

            _x_nuis_names =  tuple(
                ('x_model_uncor_shift_nuis_{}'.format(i) for i in range(fit.data_size))
            )

        _ref = dict(
            parameter_names=fit.poi_names + _x_cor_nuis_names + _y_cor_nuis_names + _x_nuis_names,
            poi_values=tuple(_ref_poi_values)
        )

        return _ref

    @parameterized.expand(PARAMETER_SPEC)
    def test_fit_parameters(self, name, fit_kwargs):

        _fit = self._make_fit(**fit_kwargs)

        # skip untestable cases
        if _fit is None:
            print('{} FOR {}'.format(
                'CHECK SKIPPED: --- ----- -- ------',
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in fit_kwargs.items()])
            ))

        _ref = self._get_ref(_fit, **fit_kwargs)

        _ref_par_vals_only = dict(poi_values=_ref.get('poi_values'))

        for i in range(len(_fit.poi_values)):
            _kwargs = dict(fit_kwargs, par=_fit.parameter_names[i])
            _par_val = _fit.poi_values[i] if not np.isnan(_fit.poi_values[i]) else 9.999
            _success = abs(_ref_par_vals_only['poi_values'][i] - _par_val) <= 0.002
            print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
                'CHECK FAIL:' if not _success else 'CHECK OK:  ',
                _ref_par_vals_only['poi_values'][i],
                '!=' if not _success else '==',
                _par_val,
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
            ))

        self._assert_fit_properties(_fit, _ref)


class TestXYFitWithCostChi2ExamplesCase(KafeFitPropertiesAssertionMixin, unittest.TestCase):

    MINIMIZER = 'scipy'

    PARAMETER_SPEC = []

    _X_ALGOS = XYFit.X_ERROR_ALGORITHMS

    for _algo in _X_ALGOS:
        _key = string_join_if((_algo,))

        _make_fit_kwargs = dict(
            x_error_algorithm=_algo,
            do_fit=True
        )

        PARAMETER_SPEC.append((_key, _make_fit_kwargs))

    def setUp(self):
        pass

    def _make_fit(self, x_error_algorithm=None, do_fit=False):
        '''construct a fit based on test parameters'''

        _fit = XYFit(
            xy_data=XYContainer(
                x_data=np.arange(16.0),
                y_data=[0.2991126116324785, 2.558050235697161, 1.2863728164289798, 3.824686039107114,
                        2.843373362329926, 5.461953737679532, 6.103072604470123, 8.166562633164254,
                        8.78250807001851, 8.311966900704014, 8.980727588512268, 11.144142620167695,
                        11.891326143534158, 12.126133797209802, 15.805993018808897, 15.3488445186788]
            ),
            cost_function=chi2,
            x_error_algorithm=x_error_algorithm,
            model_function=lambda x, a, b: a*x + b,
            minimizer=self.MINIMIZER
        )
        _fit.add_simple_error('x', 1.0)
        _fit.add_simple_error('y', 0.1)

        if do_fit:
            _fit.do_fit()

        return _fit

    def _get_ref_par_values(self, x_error_algorithm, do_fit):

        if do_fit:
            if x_error_algorithm == 'iterative linear':
                return [0.989, 0.258]
            elif x_error_algorithm == 'nonlinear':
                return [1.026, -0.015]
            elif x_error_algorithm == 'nuisance':
                return [1.025362295647366, -0.006869333562906262]  # not tested!

        return [9.999, 9.999]  # fell through -> return dummy

    def _get_ref(self, fit, x_error_algorithm, do_fit):

        algo = x_error_algorithm  # convenience

        _x_nuis_names = tuple()

        _ref_poi_values = self._get_ref_par_values(
            x_error_algorithm, do_fit)

        # x nuisance parameters are only fitted
        # if the x error algorithm is 'nuisance'
        if algo == 'nuisance':
            _x_nuis_names =  tuple(
                ('x_model_uncor_shift_nuis_{}'.format(i) for i in range(fit.data_size))
            )

        _ref = dict(
            parameter_names=fit.poi_names + _x_nuis_names,
            poi_values=tuple(_ref_poi_values)
        )

        return _ref

    @parameterized.expand(PARAMETER_SPEC)
    def test_fit_parameters(self, name, fit_kwargs):
        _fit = self._make_fit(**fit_kwargs)
        _ref = self._get_ref(_fit, **fit_kwargs)

        self._assert_fit_properties(_fit, _ref, atol=1e-4)


@unittest.skip("nuisance parameters for split errors not fully implemented yet")
class TestXYFitWithCostChi2SplitNuisance(KafeFitPropertiesAssertionMixin, unittest.TestCase):

    MINIMIZER = 'scipy'

    PARAMETER_SPEC = []

    _ERR_SPECS = ('none', 'partcor', 'fullcor', 'uncor partcor', 'uncor fullcor')

    _X_ALGOS = XYFit.X_ERROR_ALGORITHMS

    for _algo in _X_ALGOS:
        for _x_err_spec in _ERR_SPECS:
            for _y_err_spec in _ERR_SPECS:
                _key = string_join_if((_algo, 'x', _x_err_spec, 'y', _y_err_spec))

                _make_fit_kwargs = dict(
                    x_error_algorithm=_algo,
                    x_error_spec=_x_err_spec,
                    y_error_spec=_y_err_spec,
                    do_fit=True
                )

                PARAMETER_SPEC.append((_key, _make_fit_kwargs))

    def setUp(self):
        pass

    def _make_fits(self, x_error_algorithm=None, x_error_spec=None, y_error_spec=None, do_fit=False):
        '''construct a fit based on test parameters'''

        assert isinstance(x_error_spec, str)
        assert isinstance(y_error_spec, str)

        _fits = []
        for i in range(2):
            _fits.append(XYFit(
                xy_data=XYContainer(
                    x_data=[1.0, 2.0, 3.0, 4.0],
                    y_data=[2.3, 4.2, 7.5, 9.4],
                ),
                cost_function=chi2,
                x_error_algorithm=x_error_algorithm,
                model_function=lambda x, a: a*x
            ))

        for _spec, _axis in zip((x_error_spec, y_error_spec), ('x', 'y')):
            if _spec not in self._ERR_SPECS:
                raise ValueError("Unknown error specification: {}".format(_spec))

            for _fit, _fit_nuisance in zip(_fits, (False, True)):
                if 'none' in _spec:
                    pass
                if 'partcor' in _spec:
                    _fit.add_simple_error(_axis, 1.0, correlation=0.5, fit_nuisance=_fit_nuisance)
                if 'fullcor' in _spec:
                    _fit.add_simple_error(_axis, 1.0, correlation=1.0, fit_nuisance=_fit_nuisance)
                if 'uncor' in _spec:
                    _fit.add_simple_error(_axis, 1.0)

        if do_fit:
            for _i, _fit in enumerate(_fits):
                try:
                    _fit.do_fit()
                except Exception as e:
                    # fit failed -> keep fit object for testing
                    pass

        if x_error_algorithm == 'nuisance' and x_error_spec == 'partcor' and y_error_spec == 'uncor':
            pass
            #_fit._data_container.set_error_splittable()
            # print('POIVA:', _fit._nexus.get('poi_values'))
            # print('XC NP:', _fit._nexus.get('x_total_cor_shift_nuis'))
            # print('YC NP:', _fit._nexus.get('y_total_cor_shift_nuis'))
            # print('XU NP:', _fit._nexus.get('x_model_uncor_shift_nuis'))
            # print('SHIFT:', _fit._nexus.get('nuisance_shift'))
            # print('PENAL:', _fit._nexus.get('nuisance_penalty'))
            # _fit._view_nexus_graphviz(with_node_values=True)
            _fit.use_nuisance_parameters()
            # print('CALC NP:', _fit.calculate_nuisance_parameters())
            #print('CALC NP:', _fit.calculate_nuisance_parameters())
        if x_error_algorithm == 'nuisance' and x_error_spec == 'partnuisancecor' and y_error_spec == 'uncor':
            pass
            # print('POIVA:', _fit._nexus.get('poi_values'))
            # print('XC NP:', _fit._nexus.get('x_total_cor_shift_nuis'))
            # print('YC NP:', _fit._nexus.get('y_total_cor_shift_nuis'))
            # print('XU NP:', _fit._nexus.get('x_model_uncor_shift_nuis'))
            # print('SHIFT:', _fit._nexus.get('nuisance_shift'))
            # print('PENAL:', _fit._nexus.get('nuisance_penalty'))
            # print('CALC NP:', _fit.calculate_nuisance_parameters())
            #print('FIT NP:', _fit.parameter_values[1:3])
            #_fit._view_nexus_graphviz(with_node_values=True)

        return _fits

    def _get_ref_par_values(self, x_error_algorithm, x_error_spec, y_error_spec, do_fit):

        xe, ye = x_error_spec, y_error_spec  # convenience
        algo = x_error_algorithm  # convenience

        if do_fit:
            if xe == 'none':
                if ye == 'none' or ye == 'fullcor' or ye == 'fullnuisancecor' or ye == 'uncor':
                    if algo == 'iterative linear': return [2.360]
                    elif algo == 'nonlinear':      return [2.360]
                    elif algo == 'nuisance':       return [2.365]
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.400]
                    elif algo == 'nonlinear':      return [2.400]
                    elif algo == 'nuisance':       return [2.377]
            elif xe == 'uncor':
                if ye == 'none':
                    if algo == 'iterative linear': return [2.360]
                    elif algo == 'nonlinear':      return [2.366]
                    elif algo == 'nuisance':       return [2.365]
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.360]
                    elif algo == 'nonlinear':      return [2.365]
                    elif algo == 'nuisance':       return [2.365]
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.385]
                    elif algo == 'nonlinear':      return [2.377]
                    elif algo == 'nuisance':       return [2.377]
                elif ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.400]
                    elif algo == 'nonlinear':      return [2.379]
                    elif algo == 'nuisance':       return [2.365]
            elif xe == 'partcor' or xe == 'partnuisancecor':
                if ye == 'none':
                    if algo == 'iterative linear': return [2.400]
                    elif algo == 'nonlinear':      return [2.418]
                    elif algo == 'nuisance':       return [2.410]
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.385]
                    elif algo == 'nonlinear':      return [2.411]
                    elif algo == 'nuisance':       return [2.410]
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.400]
                    elif algo == 'nonlinear':      return [2.414]
                    elif algo == 'nuisance':       return [2.415]
                elif ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.417]
                    elif algo == 'nonlinear':      return [2.422]
                    elif algo == 'nuisance':       return [2.410]
            elif xe == 'fullcor' or xe == 'fullnuisancecor':
                if ye == 'none' or ye == 'fullcor' or ye == 'fullnuisancecor':
                    if algo == 'iterative linear': return [2.360]
                    elif algo == 'nonlinear':      return [2.360]
                    elif algo == 'nuisance':       return [2.365]
                elif ye == 'uncor':
                    if algo == 'iterative linear': return [2.400]
                    elif algo == 'nonlinear':      return [2.441]
                    elif algo == 'nuisance':       return [2.365]
                elif ye == 'partcor' or ye == 'partnuisancecor':
                    if algo == 'iterative linear': return [2.417]
                    elif algo == 'nonlinear':      return [2.443]
                    elif algo == 'nuisance':       return [2.377]

        return [9.999]  # fell through -> return dummy

    def _get_ref(self, fit, x_error_algorithm, x_error_spec, y_error_spec, do_fit):

        xe, ye = x_error_spec, y_error_spec  # convenience
        algo = x_error_algorithm  # convenience

        for _spec in (xe, ye):
            if _spec not in self._ERR_SPECS:
                raise ValueError("Unknown error specification: {}".format(_spec))

        _x_cor_nuis_names, _y_cor_nuis_names, _x_nuis_names = tuple(), tuple(), tuple()

        _ref_poi_values = self._get_ref_par_values(
            x_error_algorithm, x_error_spec, y_error_spec, do_fit)

        if 'nuisancecor' in xe:
            _x_cor_nuis_names= ('x_data_cor_shift_nuis_0',)

        if 'nuisancecor' in ye:
            _y_cor_nuis_names= ('y_data_cor_shift_nuis_0',)

        # x nuisance parameters are only fitted
        # if the x error algorithm is 'nuisance'
        # and the x error matrix is invertible
        if (
            algo == 'nuisance' and
            ('uncor' in xe or
             'partcor' in xe or
             'partnuisancecor' in xe)):

            _x_nuis_names =  tuple(
                ('x_model_uncor_shift_nuis_{}'.format(i) for i in range(fit.data_size))
            )

        _ref = dict(
            parameter_names=fit.poi_names + _x_cor_nuis_names + _y_cor_nuis_names + _x_nuis_names,
            poi_values=tuple(_ref_poi_values)
        )

        return _ref

    @parameterized.expand(PARAMETER_SPEC)
    def test_fit_parameters(self, name, fit_kwargs):
        #print(fit_kwargs)
        _fits = self._make_fits(**fit_kwargs)

        # skip untestable cases
        if _fits is None:
            print('{} FOR {}'.format(
                'CHECK SKIPPED: --- ----- -- ------',
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in fit_kwargs.items()])
            ))
            raise AssertionError

        _ref = self._get_ref(_fits[0], **fit_kwargs)

        #self._assert_fit_properties(_fit, _ref)

        _ref_par_vals_only = dict(poi_values=_ref.pop('poi_values'))

        # for i in range(len(_fits[0].poi_values)):
        #     _kwargs = dict(fit_kwargs, par=_fits[0].parameter_names[i])
        #     _par_val = _fits[0].poi_values[i] if not np.isnan(_fits[0].poi_values[i]) else 9.999
        #     _success = abs(_ref_par_vals_only['poi_values'][i] - _par_val) <= 0.002
        #     print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
        #         'CHECK FAIL:' if not _success else 'CHECK OK:  ',
        #         _ref_par_vals_only['poi_values'][i],
        #         '!=' if not _success else '==',
        #         _par_val,
        #         ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
        #     ))
        # raise AssertionError

        for i in range(len(_fits[0].poi_values)):
            _kwargs = dict(fit_kwargs, par=_fits[0].parameter_names[i])

            _par_val_0 = _fits[0].poi_values[i]
            _par_val_1 = _fits[1].poi_values[i]

            _par_val_0 = _par_val_0 if not np.isnan(_par_val_0) else 9.999
            _par_val_1 = _par_val_1 if not np.isnan(_par_val_1) else 9.999

            _success = abs(_par_val_0 - _par_val_1) <= 0.001
            print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
                'CHECK FAIL:' if not _success else 'CHECK OK:  ',
                _par_val_0,
                '!=' if not _success else '==',
                _par_val_1,
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
            ))

        _kwargs = dict(fit_kwargs, par='cost')

        _par_val_0 = _fits[0].cost_function_value
        _par_val_1 = _fits[1].cost_function_value

        _par_val_0 = _par_val_0 if not np.isnan(_par_val_0) else 9.999
        _par_val_1 = _par_val_1 if not np.isnan(_par_val_1) else 9.999

        _success = abs(_par_val_0 - _par_val_1) <= 0.002
        print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
            'CHECK FAIL:' if not _success else 'CHECK OK:  ',
            _par_val_0,
            '!=' if not _success else '==',
            _par_val_1,
            ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
        ))
        raise AssertionError


@unittest.skip("'nuisance' x error algorithm not fully implemented yet")
class TestXYFitWithCostChi2NuisanceFlipXY(KafeFitPropertiesAssertionMixin, unittest.TestCase):

    MINIMIZER = 'scipy'

    PARAMETER_SPEC = []

    _ERR_SPECS = ('none', 'uncor', 'partcor', 'fullcor', 'partnuisancecor', 'fullnuisancecor')

    for _x_err_spec in _ERR_SPECS:
        for _y_err_spec in _ERR_SPECS:
            _key = string_join_if(('x', _x_err_spec, 'y', _y_err_spec))

            _make_fit_kwargs = dict(
                x_error_spec=_x_err_spec,
                y_error_spec=_y_err_spec,
                do_fit=True,
            )

            PARAMETER_SPEC.append((_key, _make_fit_kwargs))

    def _make_fits(self, x_error_spec=None, y_error_spec=None, do_fit=False):
        '''construct a fit based on test parameters'''

        assert isinstance(x_error_spec, str)
        assert isinstance(y_error_spec, str)

        # -- return `None` for untestable cases

        ## 'nuisance' requires the `y` matrices to be invertible
        #if y_error_spec in ('none', 'fullcor', 'fullnuisancecor'):
        #    return None

        _fits = []
        for flip_xy in (True, False):
            _fit = XYFit(
                xy_data=XYContainer(
                    x_data=[1.0, 2.0, 3.0, 4.0] if not flip_xy else [2.3, 4.2, 7.5, 9.4],
                    y_data=[2.3, 4.2, 7.5, 9.4] if not flip_xy else [1.0, 2.0, 3.0, 4.0],
                ),
                cost_function=chi2,
                x_error_algorithm='nuisance',
                model_function=lambda x, a, b: a*x + b
            )

            for _spec, _axis in zip((x_error_spec, y_error_spec), ('x', 'y')):
                if _spec not in self._ERR_SPECS:
                    raise ValueError("Unknown error specification: {}".format(_spec))

                if 'uncor' in _spec:
                    _fit.add_simple_error(_axis, 1.0)
                if 'partcor' in _spec:
                    #_fit.add_simple_error(_axis, 1.0, correlation=0.5, fit_nuisance=False)
                    _fit.add_simple_error(_axis, 1.0, correlation=0.0, fit_nuisance=False)
                    _fit.add_simple_error(_axis, 1.0, correlation=1.0, fit_nuisance=False)

                if 'fullcor' in _spec:
                    _fit.add_simple_error(_axis, 1.0, correlation=1.0, fit_nuisance=False)
                if 'partnuisancecor' in _spec:
                    #_fit.add_simple_error(_axis, 1.0, correlation=0.5, fit_nuisance=True)
                    _fit.add_simple_error(_axis, 1.0, correlation=0.0, fit_nuisance=True)
                    _fit.add_simple_error(_axis, 1.0, correlation=1.0, fit_nuisance=True)

                    print(_axis, _spec, np.diag(np.sqrt(_fit._data_container.split_errors(_axis)[1])))
                    assert (np.diag(_fit._data_container.split_errors(_axis)[1]) == 1.0).all()
                if 'fullnuisancecor' in _spec:
                    _fit.add_simple_error(_axis, 1.0, correlation=1.0, fit_nuisance=True)

            if do_fit:
                _fit.do_fit()

            _fits.append(_fit)

        if x_error_spec == 'uncor' and y_error_spec == 'uncor':
            from kafe2.core.fitters.nexus import NodeChildrenPrinter
            _fits[1].set_poi_values((1.0/_fits[0].poi_values[0], -_fits[0].poi_values[1]/_fits[0].poi_values[0]))
            for _fit in _fits:
                NodeChildrenPrinter(_fit._nexus.get('cost')).run()
                #print('XRESID', (_fit.x_data - _fit.x_model)/_fit.y_data_error)
                #print('YRESID', (_fit.y_data - _fit.y_model)/_fit.y_data_error)
                #print('COST', (_fit.cost_function_value))
                # from kafe2 import Plot
                # import matplotlib.pyplot as plt
                # Plot(_fits).plot()
                # plt.savefig('abc.png')

        return _fits

    # def _get_ref_par_values(self, x_error_spec, y_error_spec, do_fit):
    #
    #     xe, ye = x_error_spec, y_error_spec  # convenience
    #
    #     if do_fit:
    #         if xe == 'none':
    #             if ye == 'none' or ye == 'fullcor' or ye == 'fullnuisancecor': pass  # fit not possible due to singular matrix
    #             elif ye == 'uncor': return [2.365] #[2.360]
    #             elif ye == 'partcor' or ye == 'partnuisancecor': return [2.377] #[2.4]
    #         elif xe == 'uncor':
    #             if ye == 'none':                                 return [9.999]
    #             elif ye == 'uncor':                              return [2.365]
    #             elif ye == 'partcor' or ye == 'partnuisancecor': return [2.383]
    #             elif ye == 'fullcor' or ye == 'fullnuisancecor': return [9.999]
    #         elif xe == 'partcor' or xe == 'partnuisancecor':
    #             if ye == 'none':                                 return [9.999]
    #             elif ye == 'uncor':                              return [2.374]
    #             elif ye == 'partcor' or ye == 'partnuisancecor': return [2.389]
    #             elif ye == 'fullcor' or ye == 'fullnuisancecor': return [9.999]
    #         elif xe == 'fullcor' or xe == 'fullnuisancecor':
    #             if ye == 'none':                                 return [9.999]
    #             elif ye == 'uncor':                              return [2.365] #[2.360]
    #             elif ye == 'partcor' or ye == 'partnuisancecor': return [2.377] #[2.400]
    #             elif ye == 'fullcor' or ye == 'fullnuisancecor': return [9.999]
    #
    #     return [9.999]  # fell through -> return dummy

    def _get_ref(self, fit, x_error_spec, y_error_spec, do_fit):

        xe, ye = x_error_spec, y_error_spec  # convenience

        for _spec in (xe, ye):
            if _spec not in self._ERR_SPECS:
                raise ValueError("Unknown error specification: {}".format(_spec))

        _x_cor_nuis_names, _y_cor_nuis_names, _x_nuis_names = tuple(), tuple(), tuple()

        if 'nuisancecor' in xe:
            _x_cor_nuis_names= ('x_data_cor_shift_nuis_0',)

        if 'nuisancecor' in ye:
            _y_cor_nuis_names= ('y_data_cor_shift_nuis_0',)

        # x nuisance parameters are only fitted
        # if the x error algorithm is 'nuisance'
        # and the x error matrix is invertible
        if ('uncor' in xe or
             'partcor' in xe or
             'partnuisancecor' in xe):

            _x_nuis_names =  tuple(
                ('x_model_uncor_shift_nuis_{}'.format(i) for i in range(fit.data_size))
            )

        _ref = dict(
            parameter_names=fit.poi_names + _x_cor_nuis_names + _y_cor_nuis_names + _x_nuis_names,
        )

        return _ref

    @parameterized.expand(PARAMETER_SPEC)
    def test_fit_parameters(self, name, fit_kwargs):
        #print(fit_kwargs)
        _fits = self._make_fits(**fit_kwargs)

        # skip untestable cases
        if _fits is None:
            print('{} FOR {}'.format(
                'CHECK SKIPPED: --- ----- -- ------',
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in fit_kwargs.items()])
            ))
            raise AssertionError

        #_ref = self._get_ref(_fit, **fit_kwargs)

        #self._assert_fit_properties(_fits[0], dict(poi_values=(1.0/_fits[1].poi_values[0])))

        #_ref_par_vals_only = dict(poi_values=_ref.pop('poi_values'))

        # y = a * x + b
        # x = 1 / a * y - b / a

        for i in range(len(_fits[0].poi_values)):
            _kwargs = dict(fit_kwargs, par=_fits[0].parameter_names[i])

            _par_val_0 = _fits[0].poi_values[i]
            _par_val_1 = _fits[1].poi_values[i]

            if i == 0:
                _par_val_1 = 1.0/_par_val_1
            else:
                _par_val_1 *= -_fits[0].poi_values[0]

            _par_val_0 = _par_val_0 if not np.isnan(_par_val_0) else 9.999
            _par_val_1 = _par_val_1 if not np.isnan(_par_val_1) else 9.999

            _success = abs(_par_val_0 - _par_val_1) <= 0.002
            print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
                'CHECK FAIL:' if not _success else 'CHECK OK:  ',
                _par_val_0,
                '!=' if not _success else '==',
                _par_val_1,
                ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
            ))

        _kwargs = dict(fit_kwargs, par='cost')

        _par_val_0 = _fits[0].cost_function_value
        _par_val_1 = _fits[1].cost_function_value

        _par_val_0 = _par_val_0 if not np.isnan(_par_val_0) else 9.999
        _par_val_1 = _par_val_1 if not np.isnan(_par_val_1) else 9.999

        _success = abs(_par_val_0 - _par_val_1) <= 0.002
        print('{} {:+1.3f} (ref) {} {:+1.3f} FOR {}'.format(
            'CHECK FAIL:' if not _success else 'CHECK OK:  ',
            _par_val_0,
            '!=' if not _success else '==',
            _par_val_1,
            ', '.join(['{:36s}'.format("{}={!r}".format(_k, _v)) for _k, _v in _kwargs.items()])
        ))
        raise AssertionError
