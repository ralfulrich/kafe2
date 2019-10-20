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


def make_chi2_xy_function(description=None):
    """Cost function factory. Constructs an appropriate signature"""

    _y_part = make_chi2_cost_function(
        data_name='y_data',
        model_name='y_model',
        shift_name='y_total_cor_shift',
        cov_mat_inverse_name='y_cov_mat_inverse',
        singular_behavior='ones'
    )

    _x_part = make_chi2_cost_function(
        data_name='x_data',
        model_name='x_model',
        shift_name='x_total_cor_shift',
        cov_mat_inverse_name='x_cov_mat_inverse',
        singular_behavior='ones'
    )

    _xy_cost = _x_part + _y_part + nuisance_penalty + constraints_penalty

    _xy_cost.formatter.description = description
    _xy_cost.formatter.name = "chi2"
    _xy_cost.formatter.latex_name = r"\chi^2"

    return _xy_cost


# -- cost functions

chi2 = make_chi2_xy_function(description='<standard>')

nll = make_nll_cost_function(
    data_name='y_data',
    model_name='y_model',
    log_pdf=stats.poisson.logpmf,
    use_likelihood_ratio=True
) + constraints_penalty + nuisance_penalty
nll.formatter.description = '<standard>'

# -- utility function for looking up cost function by string

get_from_string = make_getter_function_from_string({
    'chi2': chi2,
    'chisquared': chi2,
    'nll': nll,
    'negloglikelihood': nll,
})
