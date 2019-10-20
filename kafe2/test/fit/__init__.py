import abc
import numpy as np
import unittest2 as unittest
import six

__all__ = ['KafeAssertionsMixin']


class KafeAssertionsMixin:
    """extend standard test case by new assertions"""

    def _assert_compatible(self, value, reference, name=None, check_type=True, rtol=1e-3, atol=1e-6):

        _name_str = "value"
        if name is not None:
            _name_str = "value of '{}'".format(name)

        # check type identical to ref
        if check_type:
            try:
                self.assertIs(type(value), type(reference))
            except:
                # most likely array length mismatch
                print("\nCheck failed: {} "
                      "should be of type:\n\t{}\nand is of type:\n\t{}".format(
                        _name_str, type(reference), type(value)))
                raise

        # check value (almost) equal to ref
        try:
            self.assertTrue(
                np.allclose(
                    np.asarray(value),
                    np.asarray(reference),
                    rtol=rtol,
                    atol=atol,
                )
            )
        except ValueError:
            # most likely array length mismatch
            print("\nCheck failed: {} "
                  "should be:\n\t{}\nand is:\n\t{}".format(
                    _name_str, reference, value))
            raise
        except TypeError:
            # types contained in array do not support 'allclose'
            try:
                self.assertEqual(value, reference)
            except:
                print("\nCheck failed: {} "
                      "should be exactly:\n\t{}\nand is:\n\t{}".format(
                        _name_str, reference, value))
                raise
        except:

            _err_text = (
                  "\nCheck failed: {} "
                  "should be approximately:\n\t{}\n"
                  "within:\n\t{!r}\nand is:\n\t{}\n".format(
                    _name_str, reference, dict(rtol=rtol, atol=atol), value)
            )

            try:
                reference = np.asarray(reference)
                value = np.asarray(value)
            except TypeError:
                # values cannot be cast to arrays -> give up on reporting abs/rel diffs
                pass
            else:
                # report on abs/rel diffs
                _abs_diffs = np.abs(reference - value)
                _min_abs_diff, _max_abs_diff = np.min(_abs_diffs), np.max(_abs_diffs)

                _rel_diffs = np.abs(value / reference - 1)
                _min_rel_diff, _max_rel_diff = np.nanmin(_rel_diffs), np.nanmax(_rel_diffs)

                _err_text += (
                  "(abs diff between: {:g} and {:g})\n"
                  "(rel diff between: {:g} and {:g})\n").format(
                    _min_abs_diff, _max_abs_diff,
                    _min_rel_diff, _max_rel_diff
                )

            print('_'*70 + _err_text + '^'*70)
            raise
