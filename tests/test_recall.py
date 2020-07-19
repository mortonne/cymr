"""Test fitting and simulating recall."""

import numpy as np
import pandas as pd
import pytest
from psifr import fr

from cymr.fit import Recall
from cymr import fit
from cymr import parameters


class TestRecall(Recall):

    def prepare_sim(self, data, study_keys=None, recall_keys=None):
        data_study = data.loc[data['trial_type'] == 'study']
        data_recall = data.loc[data['trial_type'] == 'recall']
        merged = fr.merge_lists(data_study, data_recall)
        study = fr.split_lists(merged, 'study', ['input'])
        recalls = fr.split_lists(merged, 'recall', ['input'])
        return study, recalls

    def likelihood_subject(self, study, recalls, param_def, weights=None,
                           patterns=None):
        p = 2 - (param_def.fixed['x'] + 2) ** 2
        eps = 0.0001
        if p < eps:
            p = eps
        n = 1
        return np.log(p), n

    def generate_subject(self, study_dict, recall_dict, param_def, patterns=None, weights=None, **kwargs):
        recalls_list = [param_def.fixed['recalls']]
        return recalls_list


@pytest.fixture()
def data():
    data = pd.DataFrame({
        'subject': [1, 1, 1, 1, 1, 1,
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
                 1, 2, 1, 1, 1, 1],
        'distract': [1, 2, 3, np.nan, np.nan, np.nan,
                     3, 2, 1, np.nan, np.nan, np.nan],
    })
    return data


def test_dependent():
    param = {'Lfc': .7}
    dependent = {'Dfc': '1 - Lfc'}
    updated = parameters.set_dependent(param, dependent)
    expected = {'Lfc': .7, 'Dfc': .3}
    np.testing.assert_allclose(updated['Dfc'], expected['Dfc'])


def test_dynamic(data):
    param = {'B_distract': .2}
    dynamic = {'study': {'B_enc': 'distract * B_distract'}}
    study_data = fr.filter_data(data, 1, 1, 'study')
    study = fr.split_lists(study_data, 'raw', ['distract'])
    updated = parameters.set_dynamic(param, study, dynamic['study'])
    expected = {'B_distract': .2, 'B_enc': [np.array([.2, .4, .6])]}
    np.testing.assert_allclose(updated['B_enc'][0], expected['B_enc'][0])


def test_likelihood_subject(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.fixed = {'x': -2}
    subject_data = data.loc[data['subject'] == 1]
    logl, n = rec.likelihood_subject([], [], param_def)
    np.testing.assert_allclose(logl, np.log(2))


def test_likelihood(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.fixed = {'x': -2}
    logl, n = rec.likelihood(data, param_def)
    np.testing.assert_allclose(logl, np.log(2) + np.log(2))


def test_fit_subject(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.add_fixed(y=1)
    param_def.add_free(x=[-10, 10])
    subject_data = data.loc[data['subject'] == 1]
    param, logl, n, k = rec.fit_subject(subject_data, param_def)
    np.testing.assert_allclose(param['x'], -2, atol=0.00001)
    np.testing.assert_allclose(logl, np.log(2))
    assert k == 1


def test_fit_indiv(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.add_fixed(y=1)
    param_def.add_free(x=[-10, 10])
    results = rec.fit_indiv(data, param_def)
    np.testing.assert_allclose(results['x'].to_numpy(), [-2, -2], atol=0.0001)
    np.testing.assert_allclose(results['logl'].to_numpy(), np.log([2, 2]),
                               atol=0.0001)
    np.testing.assert_array_equal(results['k'].to_numpy(), [1, 1])


def test_generate_subject(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[(data['trial_type'] == 'study') &
                     (data['subject'] == 1)]
    param_def = parameters.Parameters()
    param_def.fixed = {'recalls': [1, 2]}
    # our "model" recalls the positions indicated in the recalls parameter
    rec_list = rec.generate_subject(study, {}, param_def)
    data_sim = fit.add_recalls(study, rec_list)
    expected = ['absence', 'hollow', 'pupil', 'hollow', 'pupil']
    assert data_sim['item'].to_list() == expected


def test_generate(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[data['trial_type'] == 'study']
    subj_param = {1: {'recalls': [1, 2]},
                  2: {'recalls': [2, 0, 1]}}
    sim = rec.generate(study, {}, subj_param_fixed=subj_param)
    expected = ['absence', 'hollow', 'pupil',
                'hollow', 'pupil',
                'fountain', 'piano', 'pillow',
                'pillow', 'fountain', 'piano']
    assert sim['item'].to_list() == expected
