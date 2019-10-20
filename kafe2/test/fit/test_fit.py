import abc
import numpy as np
import six

from kafe2.test.fit import KafeAssertionsMixin


class KafeFitPropertiesAssertionMixin(KafeAssertionsMixin):

    def _assert_fit_properties(self, fit, properties, rtol=1e-3, atol=1e-6):
        for _attr, _ref_val in properties.items():
            with self.subTest(attr=_attr):
                # retrieve property value
                _attr_val = getattr(fit, _attr)

                self._assert_compatible(
                    value=_attr_val,
                    reference=_ref_val,
                    name=_attr,
                    rtol=rtol,
                    atol=atol
                )


@six.add_metaclass(abc.ABCMeta)
class AbstractTestFit(KafeFitPropertiesAssertionMixin):

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

                self._assert_compatible(
                    value=_attr_val,
                    reference=_ref_val,
                    name=_attr,
                    rtol=rtol,
                    atol=atol
                )

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
