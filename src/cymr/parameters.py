"""Manage model parameter settings."""

from __future__ import annotations
import json
import copy

import numpy as np
from typing import Union, Any, Callable, Iterable, Optional
from numpy.typing import ArrayLike


def read_json(json_file: str) -> Parameters:
    """Read parameters from a JSON file."""
    with open(json_file, 'r') as f:
        par_dict = json.load(f)

    par = Parameters()
    if 'options' in par_dict:
        par.set_options(par_dict['options'])
    par.set_free(par_dict['free'])
    par.set_fixed(par_dict['fixed'])
    par.set_dependent(par_dict['dependent'])
    for trial_type, p in par_dict['dynamic'].items():
        par.set_dynamic(trial_type, p)
    return par


def set_dependent(
    param: dict[str, float], dependent: dict[str, str] = None
) -> dict[str, float]:
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


def set_dynamic(
    param: dict[str, float],
    list_data: dict[str, list[ArrayLike]],
    dynamic: dict[str, str],
) -> dict[str, Union[float, list[ArrayLike]]]:
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
    data_list = [{key: val[i] for key, val in list_data.items()} for i in range(n_list)]

    # merge in parameters
    for i in range(len(data_list)):
        data_list[i].update(param)
    updated: dict[str, Any] = param.copy()

    # set each dynamic parameter
    for key, expression in dynamic.items():
        updated[key] = []
        for data in data_list:
            # restrict eval functions to the numpy namespace
            val = eval(expression, np.__dict__, data)
            updated[key].append(val)
    return updated


def sample_parameters(
    sampler: dict[str, Union[float, Callable[[], float], tuple[float, float]]]
) -> dict[str, float]:
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
    """

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.fixed: dict[str, float] = {}
        self.free: dict[str, Iterable[float]] = {}
        self.dependent: dict[str, str] = {}
        self.dynamic: dict[str, dict[str, str]] = {}
        self._dynamic_names: set[str] = set()
        self._fields: list[str] = [
            'fixed',
            'free',
            'dependent',
            'dynamic',
        ]

    def __repr__(self) -> str:
        parts: dict[str, str] = {}
        for name in self._fields:
            obj = getattr(self, name)
            fields = [f'{key}: {value}' for key, value in obj.items()]
            parts[name] = '\n'.join(fields)
        s = '\n\n'.join([f'{name}:\n{f}' for name, f in parts.items()])
        return s

    def copy(self) -> Parameters:
        """Copy the parameters definition."""
        return copy.deepcopy(self)

    def to_json(self, json_file: str) -> None:
        """
        Write parameter definitions to a JSON file.

        Parameters
        ----------
        json_file : str
            Path to file to save json data.
        """
        data: dict[str, Any] = {
            'options': self.options,
            'fixed': self.fixed,
            'free': self.free,
            'dependent': self.dependent,
            'dynamic': self.dynamic,
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def set_options(self, *args: dict[str, Any], **kwargs: Any) -> None:
        """
        Set model options.

        While model parameters are stored as a dictionary of string: float
        pairs, model options may be used to store other information such
        as switching between different model variants.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_options(scope='list', recall_segment='item')
        >>> param_def.set_options({'option1': True, 'option2': False})
        """
        self.options.update(*args, **kwargs)

    def set_fixed(self, *args: dict[str, float], **kwargs: float):
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

    def set_free(
        self, *args: dict[str, Iterable[float]], **kwargs: Iterable[float]
    ) -> None:
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

    def set_dependent(self, *args: dict[str, str], **kwargs: str) -> None:
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

    def eval_dependent(self, param: dict[str, float]) -> dict[str, float]:
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

    def set_dynamic(
        self, trial_type: str, *args: dict[str, str], **kwargs: str
    ) -> None:
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

    def eval_dynamic(
        self,
        param: dict[str, Any],
        study: Optional[dict[str, list[ArrayLike]]] = None,
        recall: Optional[dict[str, list[ArrayLike]]] = None
    ) -> dict[str, Union[float, list[ArrayLike]]]:
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

    def get_dynamic(
        self, param: dict[str, Any], index: int
    ) -> dict[str, Union[float, list[ArrayLike]]]:
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
