"""Manage model parameter settings."""

import json
import copy

import numpy as np


def read_json(json_file):
    """Read parameters from a JSON file."""
    with open(json_file, 'r') as f:
        par_dict = json.load(f)

    par = Parameters()
    par.set_free(par_dict['free'])
    par.set_fixed(par_dict['fixed'])
    par.set_dependent(par_dict['dependent'])
    for trial_type, p in par_dict['dynamic'].items():
        par.set_dynamic(trial_type, p)
    for connect, p in par_dict['weights'].items():
        weight_dict = {decode_region(region): expr
                       for region, expr in p.items()}
        par.set_weights(connect, weight_dict)
    return par


def set_dependent(param, dependent=None):
    """
    Set values of dependent parameters.

    Parameters
    ----------
    param : dict of (str: float)
        Parameter values.

    dependent : dict of (str: str)
        For each dependent parameter, an expression defining that
        parameter in terms of other parameter values.

    Returns
    -------
    updated : dict of (str: float)
        Updated parameter values.
    """
    updated = param.copy()
    if dependent is not None:
        independent_param = param.copy()
        for name, expression in dependent.items():
            updated[name] = eval(expression, np.__dict__, independent_param)
    return updated


def set_dynamic(param, list_data, dynamic):
    """
    Set dynamic parameters for one trial type.

    Parameters
    ----------
    param : dict of (str: float)
        Parameter values.

    list_data : dict of list of numpy.array
        Data in list format with named fields.

    dynamic : dict of (str: str)
        Dynamic parameter definitions for one trial type. Expressions
        will be evaluated with both param keys and data keys available
        as variables. If a key exists on both param and data, the param
        value takes precedence.

    Returns
    -------
    updated : dict
        Parameter values updated to include dynamic parameters.
        Dynamic parameters are lists of numpy arrays.
    """
    # flip from dict of list to list of dict
    data_keys = list(list_data.keys())
    n_list = len(list_data[data_keys[0]])
    data_list = [
        {key: val[i] for key, val in list_data.items()}
        for i in range(n_list)
    ]

    # merge in parameters
    for i in range(len(data_list)):
        data_list[i].update(param)
    updated = param.copy()

    # set each dynamic parameter
    for key, expression in dynamic.items():
        updated[key] = []
        for data in data_list:
            # restrict eval functions to the numpy namespace
            val = eval(expression, np.__dict__, data)
            updated[key].append(val)
    return updated


def sample_parameters(sampler):
    """Randomly sample parameters."""
    param = {}
    for name, value in sampler.items():
        if callable(value):
            param[name] = value()
        elif isinstance(value, tuple):
            param[name] = value[0] + np.random.rand() * value[1]
        else:
            param[name] = value
    return param


def encode_region(region):
    """Encode a region as a string."""
    if len(region) == 0:
        raise ValueError('Cannot encode an empty region.')
    elif isinstance(region[0], str):
        region_str = '-'.join(region)
    else:
        region_str = '_'.join('-'.join(segment) for segment in region)
    return region_str


def decode_region(region_str):
    """Decode a region string."""
    if '_' in region_str:
        region = tuple([tuple(s.split('-')) for s in region_str.split('_')])
    else:
        region = tuple(region_str.split('-'))
    return region


class Parameters(object):
    """
    Class to manage model parameters.

    Attributes
    ----------
    fixed : dict of (str: float)
        Values of fixed parameters.

    free : dict of (str: tuple)
        Bounds of each free parameter.

    dependent : dict of (str: str)
        Expressions to define dependent parameters based the other
        parameters.

    dynamic : dict of (str: dict of (str: str))
        First dict specifies trial_type for dynamic parameters,
        second dict keys are parameter names, and values are
        expressions specifying how to update the parameter.

    weights : dict of (str: dict of (str: str))
        Weights template to set network connections.
    """

    def __init__(self):
        self.fixed = {}
        self.free = {}
        self.dependent = {}
        self.dynamic = {}
        self.sublayers = {}
        self.weights = {}
        self.sublayer_param = {}
        self._dynamic_names = set()

    def __repr__(self):
        names = [
            'fixed', 'free', 'dependent', 'dynamic', 'weights', 'sublayers'
        ]
        parts = {}
        for name in names:
            obj = getattr(self, name)
            fields = [f'{key}: {value}' for key, value in obj.items()]
            parts[name] = '\n'.join(fields)
        s = '\n\n'.join([f'{name}:\n{f}' for name, f in parts.items()])
        return s

    def copy(self):
        """Copy the parameters definition."""
        param = Parameters()
        param.fixed = self.fixed.copy()
        param.free = self.free.copy()
        param.dependent = self.dependent.copy()
        param.dynamic = copy.deepcopy(self.dynamic)
        param.sublayers = copy.deepcopy(self.sublayers)
        param.weights = copy.deepcopy(self.weights)
        param.sublayer_param = copy.deepcopy(self.sublayer_param)
        param._dynamic_names = self._dynamic_names.copy()
        return param

    def set_fixed(self, *args, **kwargs):
        """Set fixed parameter values."""
        self.fixed.update(*args, **kwargs)

    def set_free(self, *args, **kwargs):
        """Set free parameter ranges."""
        self.free.update(*args, **kwargs)

    def set_dependent(self, *args, **kwargs):
        """Set dependent parameters in terms of other parameters."""
        self.dependent.update(*args, **kwargs)

    def set_dynamic(self, trial_type, *args, **kwargs):
        """Set dynamic parameters in terms of parameters and data."""
        if trial_type in self.dynamic:
            self.dynamic[trial_type].update(*args, **kwargs)
        else:
            self.dynamic[trial_type] = dict(*args, **kwargs)
        for key in self.dynamic[trial_type].keys():
            self._dynamic_names.add(key)

    def set_sublayers(self, *args, **kwargs):
        """Set layers and sublayers of a network."""
        self.sublayers.update(*args, **kwargs)

    def set_sublayer_param(self, layer, *args, **kwargs):
        """Set sublayer parameters."""
        if layer in self.sublayer_param:
            self.sublayer_param[layer].update(*args, **kwargs)
        else:
            self.sublayer_param[layer] = dict(*args, **kwargs)

    def set_weights(self, connect, *args, **kwargs):
        """Set weights on model patterns."""
        if connect in self.weights:
            self.weights[connect].update(*args, **kwargs)
        else:
            self.weights[connect] = dict(*args, **kwargs)

    def eval_dependent(self, param):
        """Evaluate dependent parameters based on input parameters."""
        return set_dependent(param, self.dependent)

    def eval_dynamic(self, param, study=None, recall=None):
        """Evaluate dynamic parameters based on data fields."""
        if 'study' in self.dynamic and study is not None:
            param = set_dynamic(param, study, self.dynamic['study'])
        if 'recall' in self.dynamic and recall is not None:
            param = set_dynamic(param, recall, self.dynamic['recall'])
        return param

    def get_dynamic(self, param, index):
        """Get list-specific parameters."""
        indexed = param.copy()
        for name in self._dynamic_names:
            indexed[name] = param[name][index]
        return indexed

    def eval_sublayer_param(self, layer, param, n_trial=None):
        """Evaluate sublayer parameters."""
        eval_param = param.copy()

        # get parameter values for each sublayer
        param_lists = {}
        for sublayer in self.sublayers[layer]:
            for par, expr in self.sublayer_param[layer][sublayer].items():
                if par not in param_lists:
                    param_lists[par] = []
                value = eval(expr, np.__dict__, param)
                param_lists[par].append(value)

        # prepare parameter arrays
        for par, values in param_lists.items():
            if n_trial is not None:
                eval_param[par] = np.tile(np.asarray(values), (n_trial, 1))
            else:
                eval_param[par] = np.array(values)
        return eval_param

    def eval_weights(self, patterns, param=None, item_index=None):
        """Evaluate weights based on parameters and patterns."""
        weights = {}
        for sublayer, regions in self.weights.items():
            # get necessary patterns
            weights[sublayer] = {}
            if sublayer in ['fc', 'cf']:
                layer_type = 'vector'
            elif sublayer == 'ff':
                layer_type = 'similarity'
            else:
                raise ValueError(f'Invalid sublayer: {sublayer}.')
            data = patterns[layer_type].copy()

            # filter by item index
            if item_index is not None:
                for feature, mat in data.items():
                    if layer_type == 'vector':
                        data[feature] = mat[item_index, :]
                    else:
                        data[feature] = mat[np.ix_(item_index, item_index)]

            # evaluate expressions to get weights
            if param is not None:
                data.update(param)
            for region, expr in regions.items():
                weights[sublayer][region] = eval(expr, np.__dict__, data)
        return weights

    def to_json(self, json_file):
        """Write parameter definitions to a JSON file."""
        data = {'fixed': self.fixed, 'free': self.free,
                'dependent': self.dependent, 'dynamic': self.dynamic,
                'weights': {}}
        for layer, regions in self.weights.items():
            data['weights'][layer] = {}
            for region, expr in regions.items():
                region_str = encode_region(region)
                data['weights'][layer][region_str] = expr
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
