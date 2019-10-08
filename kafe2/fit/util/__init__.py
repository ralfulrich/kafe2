r"""
.. module:: kafe2.fit.util
    :platform: Unix
    :synopsis: This submodule provides utility functions for other modules

.. moduleauthor:: Johannes Gaessler <johannes.gaessler@student.kit.edu>
"""

import numpy as _np
from functools import partial as _partial
import itertools as _itertools

from . import function_library

# no __all__: import everything


# -- general utility functions

def string_join_if(pieces, delim='_', condition=lambda x: x):
    '''Join all elements of `pieces` that pass `condition` together
    using delimiter `delim`.'''
    return delim.join((p for p in pieces if condition(p)))


def _iter_dict(dict_, iter_func, **iter_func_kwargs):
    """Yield sequence of dicts whose keys map to the results of applying `iter_func` on the values"""
    _keys = dict_.keys()
    for instance in iter_func(*dict_.values(), **iter_func_kwargs):
        yield dict(zip(_keys, instance))


# some concrete
product_dict = _partial(_iter_dict, iter_func=_itertools.product)
zip_dict = _partial(_iter_dict, iter_func=zip)
zip_longest_dict = _partial(_iter_dict, iter_func=_itertools.zip_longest)


# -- array/matrix utility functions

def add_in_quadrature(*args):
    '''return the square root of the sum of squares of all arguments'''
    return _np.sqrt(_np.sum([_a**2 for _a in args], axis=0))


def invert_matrix(mat):
    '''perform matrix inversion'''
    return _np.linalg.inv(mat)


def maybe_invert_matrix(mat):
    '''perform matrix inversion and return None if not possible'''
    try:
        return invert_matrix(mat)
    except _np.linalg.LinAlgError:
        return None


def collect(*args):
    '''collect arguments into array'''
    return _np.asarray(args)
