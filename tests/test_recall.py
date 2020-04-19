"""Test fitting and simulating recall."""

import numpy as np
import pandas as pd
import pytest
from psifr import fr
from cymr.fit import Recall
from cymr import fit


class TestRecall(Recall):

    def prepare_fit(self, data):
        data_study = data.loc[data['trial_type'] == 'study']
        data_recall = data.loc[data['trial_type'] == 'recall']
        merged = fr.merge_lists(data_study, data_recall)
        study = fr.split_lists(merged, 'study', ['input'])
        recalls = fr.split_lists(merged, 'recall', ['input'])
        return study, recalls

    def likelihood_subject(self, study, recalls, param):
        p = 2 - (param['x'] + 2) ** 2
        eps = 0.0001
        if p < eps:
            p = eps
        return np.log(p)

    def generate_subject(self, study, param, **kwargs):
        recalls_list = [param['recalls']]
        data = fit.add_recalls(study, recalls_list)
        return data


@pytest.fixture()
def data():
    data = pd.DataFrame(
        {'subject': [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2],
         'list': [1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1],
         'trial_type': ['study', 'study', 'study',
                        'recall', 'recall', 'recall',
                        'study', 'study', 'study',
                        'recall', 'recall', 'recall'],
         'position': [1, 2, 3, 1, 2, 3,
                      1, 2, 3, 1, 2, 3],
         'item': ['absence', 'hollow', 'pupil',
                  'hollow', 'pupil', 'empty',
                  'fountain', 'piano', 'pillow',
                  'pillow', 'fountain', 'pillow'],
         'item_index': [0, 1, 2, 1, 2, np.nan,
                        3, 4, 5, 5, 3, 5],
         'task': [1, 2, 1, 2, 1, np.nan,
                  1, 2, 1, 1, 1, 1]})
    return data


def test_likelihood_subject(data):
    data = data.copy()
    rec = TestRecall()
    param = {'x': -2}
    subject_data = data.loc[data['subject'] == 1]
    logl = rec.likelihood_subject([], [], param)
    np.testing.assert_allclose(logl, np.log(2))


def test_likelihood(data):
    data = data.copy()
    rec = TestRecall()
    param = {'x': -2}
    logl = rec.likelihood(data, param)
    np.testing.assert_allclose(logl, np.log(2) + np.log(2))


def test_fit_subject(data):
    data = data.copy()
    rec = TestRecall()
    fixed = {'y': 1}
    var_names = ['x']
    var_bounds = {'x': [-10, 10]}
    subject_data = data.loc[data['subject'] == 1]
    param, logl = rec.fit_subject(subject_data, fixed, var_names, var_bounds)
    np.testing.assert_allclose(param['x'], -2, atol=0.00001)
    np.testing.assert_allclose(logl, np.log(2))


def test_fit_indiv(data):
    data = data.copy()
    rec = TestRecall()
    fixed = {'y': 1}
    var_names = ['x']
    var_bounds = {'x': [-10, 10]}
    results = rec.fit_indiv(data, fixed, var_names, var_bounds)
    np.testing.assert_allclose(results['x'].to_numpy(), [-2, -2], atol=0.0001)
    np.testing.assert_allclose(results['logl'].to_numpy(), np.log([2, 2]),
                               atol=0.0001)


def test_generate_subject(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[(data['trial_type'] == 'study') &
                     (data['subject'] == 1)]

    # our "model" recalls the positions indicated in the recalls parameter
    sim = rec.generate_subject(study, {'recalls': [1, 2]})
    expected = ['absence', 'hollow', 'pupil', 'hollow', 'pupil']
    assert sim['item'].to_list() == expected


def test_generate(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[data['trial_type'] == 'study']
    subj_param = {1: {'recalls': [1, 2]},
                  2: {'recalls': [2, 0, 1]}}
    sim = rec.generate(study, {}, subj_param)
    expected = ['absence', 'hollow', 'pupil',
                'hollow', 'pupil',
                'fountain', 'piano', 'pillow',
                'pillow', 'fountain', 'piano']
    assert sim['item'].to_list() == expected
