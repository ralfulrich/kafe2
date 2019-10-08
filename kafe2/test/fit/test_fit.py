import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractTestFit(object):

    MINIMIZER = None

    FIT_CLASS = None

    @abc.abstractmethod
    def setUp(self):
        pass

    def _assert_fit_properties(self, fit, properties, rtol=1e-3, atol=1e-6):
        for _attr, _ref_val in properties.items():
            with self.subTest(attr=_attr):
                # retrieve property value
                _attr_val = getattr(fit, _attr)

                # check type identical to ref
                self.assertIs(type(_attr_val), type(_ref_val))

                # check value (almost) equal to ref
                try:
                    self.assertTrue(
                        np.allclose(
                            np.asarray(_attr_val),
                            np.asarray(_ref_val),
                            rtol=rtol,
                            atol=atol,
                        )
                    )
                except ValueError:
                    # most likely array length mismatch
                    print("\nCheck failed: attribute {!r} "
                          "should be:\n\t{}\nand is:\n\t{}".format(
                            _attr, _ref_val, _attr_val))
                    raise
                except TypeError:
                    # types contained in array do not support 'allclose'
                    try:
                        self.assertEqual(_attr_val, _ref_val)
                    except:
                        print("\nCheck failed: attribute {!r} "
                              "should be exactly:\n\t{}\nand is:\n\t{}".format(
                                _attr, _ref_val, _attr_val))
                        raise
                except:
                    _abs_diffs = np.abs(_ref_val - _attr_val)
                    _min_abs_diff, _max_abs_diff = np.min(_abs_diffs), np.max(_abs_diffs)

                    _rel_diffs = np.abs(_attr_val/_ref_val - 1)
                    _min_rel_diff, _max_rel_diff = np.nanmin(_rel_diffs), np.nanmax(_rel_diffs)

                    print('_'*70 +
                          "\nCheck failed: attribute {!r} "
                          "should be approximately:\n\t{}\n"
                          "within:\n\t{!r}\nand is:\n\t{}\n"
                          "(abs diff between: {:g} and {:g})\n"
                          "(rel diff between: {:g} and {:g})\n".format(
                            _attr, _ref_val, dict(rtol=rtol, atol=atol), _attr_val,
                            _min_abs_diff, _max_abs_diff,
                            _min_rel_diff, _max_rel_diff) + '^'*70)
                    raise

    @abc.abstractmethod
    def _get_test_fits(self):
        pass

    def run_test_for_all_fits(self, ref_prop_dict, call_before_fit=None, fit_names=None, **kwargs):
        for _fit_name, _fit in self._get_test_fits().items():
            # skip non-requested
            if fit_names is not None and _fit_name not in fit_names:
                continue
            with self.subTest(fit=_fit_name):
                # call a user-supplied function
                if call_before_fit:
                    call_before_fit(_fit)

                # test object properties
                self._assert_fit_properties(
                    _fit,
                    ref_prop_dict,
                    **kwargs
                )

    # -- test cases to be run for *all* fit types

    def test_report_before_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.do_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_dynamic_properties(self):
        _fit = self._get_fit()
        _ref_props = {}
        _failed_props = {}
        for _p in _fit.__dynamic_props__:
            try:
                _ref_props[_p] = _fit._nexus.get(_p).value
            except np.linalg.LinAlgError as _exc:
                # expect properties to be none if nexus throws LinAlgError
                _ref_props[_p] = None
            except Exception as _exc:
                _failed_props[_p] = _exc

        # fail prematurely if there are failed properties
        for _fp_name, _fp_exc in _failed_props.items():
            with self.subTest(failed_prop=_fp_name):
                raise _fp_exc

        self._assert_fit_properties(_fit, _ref_props)
