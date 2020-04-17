"""Simulate free recall experiments."""

from abc import ABC, abstractmethod
from scipy import optimize
from psifr import fr
import pandas as pd


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
    def recall_subject(self, study_data, param):
        pass

    def generate_subject(self, study, param, **kwargs):
        recall = self.recall_subject(study, param)
        data = fr.merge_lists(study, recall, **kwargs)
        return data

    def generate(self, study, group_param, subj_param):
        subjects = study['subject'].unique()
        data_list = []
        for subject in subjects:
            param = group_param.copy()
            param.update(subj_param[subject])
            subject_study = study.loc[study['subject'] == subject]
            subject_data = self.generate_subject(subject_study, param)
            data_list.append(subject_data)
        data = pd.concat(data_list, axis=0)
        return data
