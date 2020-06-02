"""Simulate free recall experiments."""

from abc import ABC, abstractmethod
import json
import numpy as np
from scipy import optimize
import pandas as pd
from joblib import Parallel, delayed
from psifr import fr


def prepare_lists(data, study_keys=None, recall_keys=None, clean=True):
    """
    Prepare study and recall data for simulation.

    Return data information split by list. This format is similar to
    frdata structs used in EMBAM.

    Parameters
    ----------
    data : pandas.DataFrame
        Free recall data in Psifr format.

    study_keys : list of str, optional
        Columns to export for study list data. Default is:
        ['input', 'item_index']. Input position is assumed to be
        one-indexed.

    recall_keys : list of str, optional
        Columns to export for recall list data. Default is: ['input'].
        Input position is assumed to be one-indexed.

    clean : bool, optional
        If true, repeats and intrusions will be removed.

    Returns
    -------
    study : dict of (str: list of numpy.array)
        Study columns in list format.

    recall : dict of (str: list of numpy.array)
        Recall columns in list format.
    """
    if study_keys is None:
        study_keys = ['input', 'item_index']

    if recall_keys is None:
        recall_keys = ['input']

    s_keys = study_keys.copy()
    s_keys.remove('input')
    r_keys = recall_keys.copy()
    r_keys.remove('input')
    merged = fr.merge_free_recall(data, study_keys=s_keys, recall_keys=r_keys)
    if clean:
        merged = merged.query('~intrusion and repeat == 0')

    study = fr.split_lists(merged, 'study', study_keys)
    recall = fr.split_lists(merged, 'recall', recall_keys)

    for i in range(len(study['input'])):
        if 'input' in study_keys:
            study['input'][i] = study['input'][i].astype(int) - 1
        if 'item_index' in study_keys:
            study['item_index'][i] = study['item_index'][i].astype(int)

        if 'input' in recall_keys:
            recall['input'][i] = recall['input'][i].astype(int) - 1
        if 'item_index' in recall_keys:
            recall['item_index'][i] = recall['item_index'][i].astype(int)

    n = np.unique([len(items) for items in study['input']])
    if len(n) > 1:
        raise ValueError('List length must not vary.')
    return study, recall


def prepare_study(study_data, study_keys=None):
    """
    Prepare study phase data for simulation.

    Parameters
    ----------
    study_data : pandas.DataFrame
        Study list data. Position is assumed to be one-indexed.

    study_keys : list of str
        Columns to export to split list format.

    Returns
    -------
    study : dict of (str: numpy.array)
        Study columns in split list format.
    """
    if study_keys is None:
        study_keys = ['position', 'item_index']

    study = fr.split_lists(study_data, 'raw', study_keys)
    for i in range(len(study['position'])):
        if 'position' in study_keys:
            study['position'][i] = study['position'][i].astype(int) - 1
        if 'item_index' in study_keys:
            study['item_index'][i] = study['item_index'][i].astype(int)
    return study


def add_recalls(study, recalls_list):
    """
    Add recall sequences to a study DataFrame.

    Parameters
    ----------
    study : pandas.DataFrame
        Study list data.

    recalls_list : list of list of int
        Recall sequence for each list in output order. Each entry is
        the index of the recalled item in the list.

    Returns
    -------
    data : pandas.DataFrame
        Complete free recall DataFrame suitable for analysis.
    """
    lists = study['list'].unique()
    subjects = study['subject'].unique()
    if len(subjects) > 1:
        raise ValueError('Unpacking multiple subjects not supported.')
    subject = subjects[0]

    # initialize recall trials DataFrame
    n_recall = np.sum([len(r) for r in recalls_list])
    recall = pd.DataFrame({'subject': subject,
                           'list': np.zeros(n_recall, dtype=int),
                           'trial_type': 'recall',
                           'position': np.zeros(n_recall, dtype=int),
                           'item': ''})

    # set basic information (list, item, position)
    n = 0
    for i, seq in enumerate(recalls_list):
        pool = study.loc[study['list'] == lists[i], 'item'].to_numpy()
        for j, pos in enumerate(seq):
            recall.loc[n, 'list'] = lists[i]
            recall.loc[n, 'item'] = pool[pos]
            recall.loc[n, 'position'] = j + 1
            n += 1
    data = pd.concat((study, recall), axis=0, ignore_index=True)
    data = data.sort_values(['list', 'trial_type'], ascending=[True, False])
    return data


def get_best_results(results):
    """Get best results from a repeated search."""
    df = []
    subjects = results.index.get_level_values('subject').unique()
    for subject in subjects:
        res = results.loc[subject].reset_index()
        subject_best = res.loc[[res['logl'].argmax()]]
        df.append(subject_best)
    best = pd.concat(df, axis=0)
    best.index = subjects
    best.index.rename('subject', inplace=True)
    return best


def set_dependent(param, dependent=None):
    updated = param.copy()
    if dependent is not None:
        independent_param = param.copy()
        for name, expression in dependent.items():
            updated[name] = eval(expression, None, independent_param)
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

    weights : dict of (str: dict of (str: str))
        Weights template to set network connections.
    """

    def __init__(self):
        self.fixed = {}
        self.free = {}
        self.dependent = {}
        self.weights = {}

    def __repr__(self):
        names = ['fixed', 'free', 'dependent', 'weights']
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

    def add_weights(self, connect, *args, **kwargs):
        if connect in self.weights:
            self.weights[connect].update(*args, **kwargs)
        else:
            self.weights[connect] = dict(*args, **kwargs)

    def to_json(self, json_file):
        data = {'fixed': self.fixed, 'free': self.free,
                'dependent': self.dependent, 'weights': self.weights}
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


class Recall(ABC):
    """
    Base class for evaluating a model of free recall.

    Common Parameters
    -----------------
    study : pandas.DataFrame
        Study list information.

    recall : pandas.DataFrame
        Recall period information for each list.

    param : dict
        Model parameter values.

    patterns : dict
        May include keys: 'vector' and/or 'similarity'. Vectors are
        used to set distributed model representations. Similarity
        matrices are used to set item connections. Vector and
        similarity values are dicts of (feature: array) specifying
        an array for one or more named features, with an
        [items x units] array for vector representations, or
        [items x items] for similarity matrices.

    weights : dict
        Keys indicate which model connections to apply weighting
        to. Values are dicts of (feature: w), where w is the scale
        to apply to a given feature.
    """

    @abstractmethod
    def likelihood_subject(self, study, recall, param, patterns=None,
                           weights=None):
        """
        Log likelihood of data for one subject based on a given model.

        Parameters
        ----------
        study : dict of (str: list of numpy.array)
            Information about the study phase in list format.

        recall : dict of (str: list of numpy.array)
            Information about recalled items in list format.

        param : dict of (str: float)
            Model parameter values.

        patterns : dict of (str: dict of (str: numpy.array))
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float))
            Weights to apply to model feature patterns.

        Returns
        -------
        logl : float
            Total log likelihood of data for this subject.

        n : int
            Number of evaluated data points.
        """
        pass

    def likelihood(self, data, group_param, subj_param=None, patterns=None,
                   weights=None):
        """
        Log likelihood summed over all subjects.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to fit. Must include a 'subject' column.

        group_param : dict of (str: float)
            Values of parameters that apply to all subjects.

        subj_param : dict of (str: dict of (str: float))
            Parameters that vary by subject, indexed by subject.

        patterns : dict of (str: dict of (str: numpy.array))
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float))
            Weights to apply to model feature patterns.

        Returns
        -------
        logl : float
            Log likelihood summed over all subjects.

        n : int
            Number of evaluated data points.
        """
        subjects = data['subject'].unique()
        logl = 0
        n = 0
        for subject in subjects:
            param = group_param.copy()
            if subj_param is not None:
                param.update(subj_param[subject])
            subject_data = data.loc[data['subject'] == subject]
            study, recall = self.prepare_sim(subject_data)
            subject_logl, subject_n = self.likelihood_subject(
                study, recall, param, patterns=patterns, weights=weights)
            logl += subject_logl
            n += subject_n
        return logl, n

    @abstractmethod
    def prepare_sim(self, subject_data):
        """
        Prepare data for simulation.

        Exporting data in DataFrame format to list format for
        simulation takes time, so this is only done once before running
        a parameter search.

        Parameters
        ----------
        subject_data : pandas.DataFrame
            Data for one subject.

        Returns
        -------
        study : dict of (str: list of numpy.array)
            Information about the study phase in list format.

        recall : dict of (str: list of numpy.array)
            Information about recalled items in list format.
        """
        pass

    def fit_subject(self, subject_data, fixed, free, dependent=None,
                    patterns=None, weights=None, method='de', **kwargs):
        """
        Fit a model to data for one subject.

        Parameters
        ----------
        subject_data : pandas.DataFrame
            Data for one subject.

        fixed : dict of (str: float)
            Values of fixed parameters.

        free : dict of (str: (float, float))
            Allowed range of free parameters.

        dependent : dict of (str: str), optional
            Expressions to evaluate to set dependent parameters.

        patterns : dict of (str: dict of (str: numpy.array))
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float))
            Weights to apply to model feature patterns.

        method : str, optional
            Search method for fitting the parameters.

        kwargs
            Additional keyword arguments for the search method.

        Returns
        -------
        param : dict of (str: float)
            Best-fitting parameters, including fixed, free, and
            dependent parameters.

        logl : float
            Log likelihood for the best-fitting parameters.

        n : int
            Number of data points evaluated.

        k : int
            Number of free parameters.
        """
        study, recall = self.prepare_sim(subject_data)
        var_names = list(free.keys())

        def eval_fit(x):
            eval_param = fixed.copy()
            eval_param.update(dict(zip(var_names, x)))
            eval_param = set_dependent(eval_param, dependent)
            eval_logl, _ = self.likelihood_subject(study, recall, eval_param,
                                                   patterns, weights)
            return -eval_logl

        group_lb = [free[k][0] for k in var_names]
        group_ub = [free[k][1] for k in var_names]
        bounds = optimize.Bounds(group_lb, group_ub)
        if method == 'de':
            res = optimize.differential_evolution(eval_fit, bounds, **kwargs)
        elif method == 'shgo':
            b = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
            res = optimize.shgo(eval_fit, b, **kwargs)
        else:
            raise ValueError(f'Invalid method: {method}')

        # fitted parameters
        param = fixed.copy()
        param.update(dict(zip(var_names, res['x'])))
        param = set_dependent(param, dependent)

        # evaluate fitted parameters, get number of fitted points
        logl, n = self.likelihood_subject(study, recall, param,
                                          patterns, weights)
        k = len(free)
        assert logl == -res['fun']
        return param, logl, n, k

    def _run_fit_subject(self, data, subject, fixed, free, dependent,
                         patterns=None, weights=None, method='de', **kwargs):
        """Apply fitting to one subject."""
        subject_data = data.loc[data['subject'] == subject]
        param, logl, n, k = self.fit_subject(
            subject_data, fixed, free, dependent, patterns, weights,
            method, **kwargs)
        results = {**param, 'logl': logl, 'n': n, 'k': k}
        return results

    def fit_indiv(self, data, fixed, free, dependent=None, patterns=None,
                  weights=None, n_jobs=None, method='de', n_rep=1, **kwargs):
        """
        Fit parameters to individual subjects.

        Parameters
        ----------
        data : pandas.DataFrame
            Data for one or more subjects.

        fixed : dict of (str: float)
            Values of fixed parameters.

        free : dict of (str: (float, float))
            Allowed range of free parameters.

        dependent : dict of (str: str), optional
            Expressions to evaluate to set dependent parameters.

        patterns : dict of (str: dict of (str: numpy.array)), optional
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float)), optional
            Weights to apply to model feature patterns.

        n_jobs : int, optional
            Number of processes to use for fitting subjects in
            parallel.

        method : str, optional
            Search method for fitting the parameters.

        n_rep : int, optional
            Number of times to repeat each search.

        kwargs
            Additional keyword arguments for the search method.

        Returns
        -------
        results : pandas.DataFrame
            Best-fitting parameters, log likelihood (:code:`logl`),
            number of data points (:code:`n`), and number of free
            parameters (:code:`k`) for each subject.
        """
        subjects = data['subject'].unique()
        full_subjects = np.repeat(subjects, n_rep)
        full_reps = np.tile(np.arange(n_rep), len(subjects))
        full_results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_fit_subject)(
                data, subject, fixed, free, dependent, patterns,
                weights, method, **kwargs
            ) for subject in full_subjects
        )
        d = {(subject, rep): res for subject, rep, res in
             zip(full_subjects, full_reps, full_results)}
        results = pd.DataFrame(d).T
        results.index.rename(['subject', 'rep'], inplace=True)
        return results

    @abstractmethod
    def generate_subject(self, study, param, patterns=None, weights=None,
                         **kwargs):
        """
        Generate simulated data for one subject.

        Parameters
        ----------
        study : dict of (str: list of numpy.array)
            Information about the study phase in list format.

        param : dict of (str: float)
            Model parameter values.

        patterns : dict of (str: dict of (str: numpy.array)), optional
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float)), optional
            Weights to apply to model feature patterns.
        """
        pass

    def generate(self, study, group_param, subj_param=None, patterns=None,
                 weights=None, n_rep=1):
        """
        Generate simulated data for all subjects.

        Parameters
        ----------
        study : dict of (str: list of numpy.array)
            Information about the study phase in list format.

        group_param : dict of (str: float)
            Values of parameters that apply to all subjects.

        subj_param : dict of (str: dict of (str: float))
            Parameters that vary by subject, indexed by subject.

        patterns : dict of (str: dict of (str: numpy.array))
            Patterns to use in the model.

        weights : dict of (str: dict of (str: float))
            Weights to apply to model feature patterns.

        n_rep : int
            Number of times to repeat the simulation for each subject.

        Returns
        -------
        data : pandas.DataFrame
            Simulated data for each subject.
        """
        subjects = study['subject'].unique()
        data_list = []
        for subject in subjects:
            param = group_param.copy()
            if subj_param is not None:
                param.update(subj_param[subject])
            subject_study = study.loc[study['subject'] == subject]
            max_list = subject_study['list'].max()
            for i in range(n_rep):
                subject_data = self.generate_subject(subject_study, param,
                                                     patterns, weights)
                subject_data['list'] = i * max_list + subject_data['list']
                data_list.append(subject_data)
        data = pd.concat(data_list, axis=0, ignore_index=True)
        return data

    def _run_parameter_recovery(self, study, fixed, free,
                                dependent=None, patterns=None, weights=None,
                                method='de', n_rep=1, **kwargs):
        """Run a parameter recovery test."""
        # generate parameters
        param = fixed.copy()
        sampled = sample_parameters(free)
        param.update(sampled)
        param = set_dependent(param, dependent)

        # generate simulated data
        sim = self.generate(study, param, patterns=patterns,
                            weights=weights, n_rep=n_rep)

        # fit the simulated data
        fitted_param, logl, n, k = self.fit_subject(
            sim, fixed, free, dependent=dependent, patterns=patterns,
            weights=weights, method=method, **kwargs
        )

        # store results
        df_sim = pd.DataFrame(param, index=['sim'])
        df_fit = pd.DataFrame(fitted_param, index=['fit'])
        df_sample = pd.concat((df_sim, df_fit), axis=0)
        return df_sample

    def parameter_recovery(self, study, n_sample, fixed, free,
                           dependent=None, patterns=None, weights=None,
                           method='de', n_rep=1, n_jobs=None, **kwargs):
        """Run multiple iterations of parameter recovery."""
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(self._run_parameter_recovery)(
                study, fixed, free, dependent, patterns,
                weights, method, n_rep, **kwargs
            ) for i in range(n_sample)
        )
        results = pd.concat(results_list, axis=0, keys=np.arange(n_sample))
        return results

    def parameter_sweep(self, study, fixed, param_names, param_sweeps,
                        dependent=None, patterns=None, weights=None, n_rep=1):
        """Simulate data with varying parameters."""
        index = pd.MultiIndex.from_product(param_sweeps, names=param_names)
        df_list = []
        for idx in index:
            param = fixed.copy()
            param.update(dict(zip(param_names, idx)))
            param = set_dependent(param, dependent)
            sim = self.generate(study, param, None, patterns=patterns,
                                weights=weights, n_rep=n_rep)
            df_list.append(sim)
        results = pd.concat(df_list, axis=0, keys=index)
        results = results.droplevel(len(param_sweeps))
        results.index.rename(param_names, inplace=True)
        return results
