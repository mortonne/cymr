"""Test fitting and simulating recall."""

import numpy as np
import pandas as pd
import pytest
from cymr.recall import Recall
from cymr import recall


class TestRecall(Recall):

    def likelihood_subject(self, data, param):
        subject = data['subject'].to_numpy()[0]
        p = subject + 1 - (param['x'] + 2) ** 2
        eps = 0.0001
        if p < eps:
            p = eps
        return np.log(p)

    def recall_subject(self, study, param, **kwargs):
        recalls_list = [[1, 2]]
        data = recall.add_recalls(study, recalls_list)
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
    logl = rec.likelihood_subject(subject_data, param)
    np.testing.assert_allclose(logl, np.log(2))


def test_likelihood(data):
    data = data.copy()
    rec = TestRecall()
    param = {'x': -2}
    logl = rec.likelihood(data, param)
    np.testing.assert_allclose(logl, np.log(2) + np.log(3))


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
    np.testing.assert_allclose(results['logl'].to_numpy(), np.log([2, 3]),
                               atol=0.0001)


def test_recall_subject(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[(data['trial_type'] == 'study') &
                     (data['subject'] == 1)]
    sim = rec.recall_subject(study, {})
    expected = ['absence', 'hollow', 'pupil', 'hollow', 'pupil']
    assert sim['item'].to_list() == expected
