import six

from ...core.fitters import NexusFitter
from ...core.fitters.nexus import Nexus, Parameter, Alias, Function
from .._base import FitBase


class MultiFit(object):
    def __init__(self, fit_objects, minimizer=None, minimizer_kwargs=None):
        self._fits = []
        for _fit in fit_objects:
            if not isinstance(_fit, FitBase):
                raise TypeError(
                    "Cannot use object of type '{}' in MultiFit: "
                    "expected type derived from `FitBase`!".format(
                        type(_fit)
                    )
                )
            self._fits.append(_fit)

        self._nexus = None
        self._fitter = None

        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

    def _init_nexus(self):
        self._nexus = Nexus()

        _nexuses = [_f._get_nexus() for _f in self._fits]

        _cost_nexus_nodes = []
        _cost_nexus_node_pairs = []
        _parameter_nodes = {}
        for _fit, _nexus in zip(self._fits, _nexuses):
            _cost_nexus_nodes.append(_nexus.get('cost'))
            _cost_nexus_node_pairs.append((_nexus, _nexus.get('cost')))
            for _par_name in _fit.parameter_names:
                _parameter_nodes.setdefault(_par_name, []).append(
                    (_nexus, _nexus.get(_par_name))
                )

        for _par_name, _par_nexus_node_pairs in six.iteritems(_parameter_nodes):
            # create new parameter node in the new (multi-)nexus
            _multi_par_node = self._nexus.add(
                Parameter(_par_nexus_node_pairs[0][1].value, name=_par_name)
            )
            # rewire the old parameters to point to the new one
            for _nexus, _node in _par_nexus_node_pairs:
                _nexus.add(
                    Alias(ref=_multi_par_node, name=_node.name),
                    existing_behavior='replace'
                )

        # create new cost parameter node in the new (multi-)nexus
        _multi_cost_node = self._nexus.add(
            Function(
                lambda *args: sum(args),
                name='multicost',
                parameters=_cost_nexus_nodes,
            )
        )

    def _init_fit_parameters(self):
        _fit_param_names = []
        for _fit in self._fits:
            for _par in _fit.parameter_names:
                if _par not in _fit_param_names:
                    _fit_param_names.append(_par)

        self._fit_param_names = _fit_param_names

    def _initialize_fitter(self, minimizer=None, minimizer_kwargs=None):
        # save minimizer, minimizer_kwargs for serialization
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._fitter = NexusFitter(nexus=self._nexus,
                                   parameters_to_fit=self._fit_param_names,
                                   parameter_to_minimize='multicost',
                                   minimizer=minimizer,
                                   minimizer_kwargs=minimizer_kwargs)

    def do_fit(self):

        if not self._nexus:
            self._init_nexus()

        self._init_fit_parameters()  # ensure parameters are up to date

        if not self._fitter:
            self._initialize_fitter(
                self._minimizer, self._minimizer_kwargs)

        self._fitter.do_fit()
