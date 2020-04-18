"""Simulate free recall experiments."""

from abc import ABC, abstractmethod
import numpy as np
from scipy import optimize
import pandas as pd


def add_recalls(study, recalls_list):
    """Add recall sequences to a study DataFrame."""
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
    return data


class Recall(ABC):

    @abstractmethod
    def likelihood_subject(self, subject_data, param):
        pass

    def likelihood(self, data, group_param, subj_param=None):
        subjects = data['subject'].unique()
        logl = 0
        for subject in subjects:
            param = group_param.copy()
            if subj_param is not None:
                param.update(subj_param[subject])
            subject_data = data.loc[data['subject'] == subject]
            subject_logl = self.likelihood_subject(subject_data, param)
            logl += subject_logl
        return logl

    def fit_subject(self, subject_data, fixed, var_names, var_bounds, **kwargs):

        def eval_fit(x):
            eval_param = fixed.copy()
            eval_param.update(dict(zip(var_names, x)))
            eval_logl = self.likelihood_subject(subject_data, eval_param)
            return -eval_logl

        group_lb = [var_bounds[k][0] for k in var_names]
        group_ub = [var_bounds[k][1] for k in var_names]
        bounds = optimize.Bounds(group_lb, group_ub)
        res = optimize.differential_evolution(eval_fit, bounds, **kwargs)

        # fitted parameters
        param = fixed.copy()
        param.update(dict(zip(var_names, res['x'])))

        logl = -res['fun']
        return param, logl

    def fit_indiv(self, data, fixed, var_names, var_bounds, **kwargs):
        subjects = data['subject'].unique()
        results = {}
        for subject in subjects:
            subject_data = data.loc[data['subject'] == subject]
            param, logl = self.fit_subject(subject_data, fixed, var_names,
                                           var_bounds, **kwargs)
            results[subject] = {**param, 'logl': logl}
        return pd.DataFrame(results).T

    @abstractmethod
    def generate_subject(self, study, param, **kwargs):
        pass

    def generate(self, study, group_param, subj_param=None):
        subjects = study['subject'].unique()
        data_list = []
        for subject in subjects:
            param = group_param.copy()
            if subj_param is not None:
                param.update(subj_param[subject])
            subject_study = study.loc[study['subject'] == subject]
            subject_data = self.generate_subject(subject_study, param)
            data_list.append(subject_data)
        data = pd.concat(data_list, axis=0, ignore_index=True)
        return data
