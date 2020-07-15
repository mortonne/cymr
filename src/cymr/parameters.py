"""Manage model parameter settings."""

import json

import numpy as np


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
            updated[name] = eval(expression, None, independent_param)
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
        self.weights = {}

    def __repr__(self):
        names = ['fixed', 'free', 'dependent', 'dynamic', 'weights']
        parts = {}
        for name in names:
            obj = getattr(self, name)
            fields = [f'{key}: {value}' for key, value in obj.items()]
            parts[name] = '\n'.join(fields)
        s = '\n\n'.join([f'{name}:\n{f}' for name, f in parts.items()])
        return s

    def add_fixed(self, *args, **kwargs):
        self.fixed.update(*args, **kwargs)

    def add_free(self, *args, **kwargs):
        self.free.update(*args, **kwargs)

    def add_dependent(self, *args, **kwargs):
        self.dependent.update(*args, **kwargs)

    def add_dynamic(self, *args, **kwargs):
        self.dynamic.update(*args, **kwargs)

    def add_weights(self, connect, *args, **kwargs):
        if connect in self.weights:
            self.weights[connect].update(*args, **kwargs)
        else:
            self.weights[connect] = dict(*args, **kwargs)

    def to_json(self, json_file):
        data = {'fixed': self.fixed, 'free': self.free,
                'dependent': self.dependent, 'dynamic': self.dynamic,
                'weights': self.weights}
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
