import numpy as np

from .._base import (
    CostFunction, make_getter_function_from_string,
    constraints_penalty, nuisance_penalty,
)

__all__ = [
    "nll",
    "get_from_string"
]


# -- cost functions

@CostFunction
def nll(model):
    return -2.0 * np.sum(np.log(np.asarray(model)))


nll = nll + constraints_penalty + nuisance_penalty
nll.formatter.description = '<standard>'


# -- utility function for looking up cost function by string

get_from_string = make_getter_function_from_string({
    'nll': nll,
    'negloglikelihood': nll,
})
