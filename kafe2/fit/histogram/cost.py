from scipy import stats

from .._base import (
    make_chi2_cost_function, make_nll_cost_function,
    make_getter_function_from_string,
    constraints_penalty, nuisance_penalty
)


__all__ = [
    "chi2",
    "nll",
    "get_from_string"
]

# -- cost functions

chi2 = make_chi2_cost_function(
    data_name='data',
    model_name='model',
    cov_mat_inverse_name='cov_mat_inverse',
    description='<standard>'
) + nuisance_penalty + constraints_penalty

nll = make_nll_cost_function(
    data_name='data',
    model_name='model',
    log_pdf=stats.poisson.logpmf,
    use_likelihood_ratio=True,
    description='<standard>'
) + nuisance_penalty + constraints_penalty

# -- utility function for looking up cost function by string

get_from_string = make_getter_function_from_string({
    'chi2': chi2,
    'chisquared': chi2,
    'nll': nll,
    'negloglikelihood': nll,
})
