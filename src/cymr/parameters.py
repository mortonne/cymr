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
    par.set_sublayers(par_dict['sublayers'])
    for connect, p in par_dict['weights'].items():
        weight_dict = {decode_region(region): expr
                       for region, expr in p.items()}
        par.set_weights(connect, weight_dict)
    for layer, sublayer_param in par_dict['sublayer_param'].items():
        for sublayer, param in sublayer_param.items():
            par.set_sublayer_param(layer, sublayer, param)
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
        for name, expression in dependent.items():
            updated[name] = eval(expression, np.__dict__, updated)
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

    sublayers : dict of (list of str)
        Names of sublayers for each layer in the network.

    weights : dict of (tuple of (tuple of str)): str
        Weights template to set network connections. Weights are
        indicated by region within the network. Each region is
        specified with a tuple giving names of sublayers and segments:
        ((f_sublayer, f_segment), (c_sublayer, c_segment)). The value
        for each region should be an expression to be evaluated with
        patterns and/or parameters.

    sublayer_param : dict of (str: dict of (str: dict of str))
        Parameters that vary by sublayer. These parameters are
        specified in terms of their layer and sublayer. Each value
        should contain an expression to be evaluated with parameters.
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
            'fixed', 'free', 'dependent', 'dynamic', 'sublayers', 'weights',
            'sublayer_param',
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
        param = type(self).__new__(self.__class__)
        param.fixed = self.fixed.copy()
        param.free = self.free.copy()
        param.dependent = self.dependent.copy()
        param.dynamic = copy.deepcopy(self.dynamic)
        param.sublayers = copy.deepcopy(self.sublayers)
        param.weights = copy.deepcopy(self.weights)
        param.sublayer_param = copy.deepcopy(self.sublayer_param)
        param._dynamic_names = self._dynamic_names.copy()
        return param

    def to_json(self, json_file):
        """
        Write parameter definitions to a JSON file.

        Parameters
        ----------
        json_file : str
            Path to file to save json data.
        """
        data = {
            'fixed': self.fixed,
            'free': self.free,
            'dependent': self.dependent,
            'dynamic': self.dynamic,
            'sublayers': self.sublayers,
            'weights': {},
            'sublayer_param': self.sublayer_param,
        }
        for layer, regions in self.weights.items():
            data['weights'][layer] = {}
            for region, expr in regions.items():
                region_str = encode_region(region)
                data['weights'][layer][region_str] = expr
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def set_fixed(self, *args, **kwargs):
        """
        Set fixed parameter values.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_fixed(a=1, b=2)
        >>> param_def.set_fixed({'c': 3, 'd': 4})
        """
        self.fixed.update(*args, **kwargs)

    def set_free(self, *args, **kwargs):
        """
        Set free parameter ranges.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_free(a=[0, 1], b=[1, 10])
        >>> param_def.set_free({'c': [3, 4], 'd': [0, 10]})
        """
        self.free.update(*args, **kwargs)

    def set_dependent(self, *args, **kwargs):
        """
        Set dependent parameters in terms of other parameters.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_dependent(a='exp(b * 2)')
        >>> param_def.set_dependent({'b': 'a * c', 'd': '2 * c + 1'})
        """
        self.dependent.update(*args, **kwargs)

    def eval_dependent(self, param):
        """
        Evaluate dependent parameters based on input parameters.

        Parameters
        ----------
        param : dict of str: float
            Parameters to use when evaluating dependent parameters.

        Returns
        -------
        eval_param : dict of str: float
            Input parameters with dependent parameters set.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_dependent(b='clip(3 * a, 0, 1)')
        >>> param_def.eval_dependent({'a': 0.5})
        {'a': 0.5, 'b': 1.0}
        """
        return set_dependent(param, self.dependent)

    def set_dynamic(self, trial_type, *args, **kwargs):
        """
        Set dynamic parameters in terms of parameters and data.

        Parameters
        ----------
        trial_type : str
            Type of trial that the parameter will vary over.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_dynamic('study', a='b * input')
        >>> param_def.set_dynamic('recall', {'c': 'd * op'})
        """
        if trial_type in self.dynamic:
            self.dynamic[trial_type].update(*args, **kwargs)
        else:
            self.dynamic[trial_type] = dict(*args, **kwargs)
        for key in self.dynamic[trial_type].keys():
            self._dynamic_names.add(key)

    def eval_dynamic(self, param, study=None, recall=None):
        """
        Evaluate dynamic parameters based on data fields.

        Parameters
        ----------
        param : dict of str: float
            Parameters to use when evaluating dynamic parameters.

        study : dict of list of numpy.array, optional
            Study data to use when evaluating parameters.

        recall : dict of list of numpy.array, optional
            Recall data to use when evaluating parameters.

        Returns
        -------
        eval_param : dict of str: float
            Input parameters with dynamic parameters set.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_dynamic('study', a='b * input')
        >>> data = {'input': [np.array([1, 2, 3])]}
        >>> param_def.eval_dynamic({'b': 0.2}, study=data)
        {'b': 0.2, 'a': [array([0.2, 0.4, 0.6])]}
        """
        if 'study' in self.dynamic and study is not None:
            param = set_dynamic(param, study, self.dynamic['study'])
        if 'recall' in self.dynamic and recall is not None:
            param = set_dynamic(param, recall, self.dynamic['recall'])
        return param

    def get_dynamic(self, param, index):
        """
        Get list-specific parameters.

        Parameters
        ----------
        param : dict
            Parameters to index.

        index : int
            Index of the list.

        Returns
        -------
        indexed : dict
            Copy of input parameters with dynamic parameters for the
            specified list.
        """
        indexed = param.copy()
        for name in self._dynamic_names:
            indexed[name] = param[name][index]
        return indexed

    def set_sublayers(self, *args, **kwargs):
        """
        Set layers and sublayers of a network.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_sublayers(f=['task'], c=['task'])
        """
        self.sublayers.update(*args, **kwargs)

    def set_weights(self, connect, regions):
        """
        Set weights on model patterns.

        Parameters
        ----------
        connect : str
            Network connection to set weights for.

        regions : dict of (tuple of (tuple of str)): str
            Weights for each region to set.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> region = (('f_sub', 'f_reg'), ('c_sub', 'c_reg'))
        >>> param_def.set_weights('fc', {region: 'b * pattern'})
        """
        if connect in self.weights:
            self.weights[connect].update(regions)
        else:
            self.weights[connect] = regions

    def eval_weights(self, patterns, param=None, item_index=None):
        """
        Evaluate weights based on parameters and patterns.

        Parameters
        ----------
        patterns : dict of str: (dict of str: numpy.ndarray)
            Patterns to use when evaluating weights.

        param : dict, optional
            Parameters to use when evaluating weights.

        item_index : numpy.ndarray, optional
            Item indices to include in the patterns.

        Returns
        -------
        weights : dict of str: (dict of str: numpy.ndarray)
            Weight matrices for each region in each connection matrix.
        """
        weights = {}
        for connect, regions in self.weights.items():
            # get necessary patterns
            weights[connect] = {}
            if connect in ['fc', 'cf']:
                layer_type = 'vector'
            elif connect == 'ff':
                layer_type = 'similarity'
            else:
                raise ValueError(f'Invalid connection: {connect}.')
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
                weights[connect][region] = eval(expr, np.__dict__, data)
        return weights

    def set_sublayer_param(self, layer, sublayer, *args, **kwargs):
        """
        Set sublayer parameters.

        Parameters
        ----------
        layer : str
            Layer containing the sublayer to set.

        sublayer : str
            Sublayer to set parameters for.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_sublayer_param('c', 'sub1', a=1})
        >>> param_def.set_sublayer_param('c', 'sub1', {'b': 2})
        """
        if layer in self.sublayer_param:
            if sublayer in self.sublayer_param[layer]:
                self.sublayer_param[layer][sublayer].update(*args, **kwargs)
            else:
                self.sublayer_param[layer][sublayer] = dict(*args, **kwargs)
        else:
            self.sublayer_param[layer] = {sublayer: dict(*args, **kwargs)}

    def eval_sublayer_param(self, layer, param, n_trial=None):
        """
        Evaluate sublayer parameters.

        Parameters
        ----------
        layer : str
            Layer to evaluate.

        param : dict
            Parameters to use when evaluating sublayer parameters.

        n_trial : int, optional
            Number of trials. If indicated, parameters will be tiled
            over all trials.

        Returns
        -------
        eval_param : dict
            Input parameters with sublayer-specific parameters set.
        """
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
