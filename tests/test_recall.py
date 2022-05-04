"""Test fitting and simulating recall."""

import numpy as np
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

    def likelihood_subject(self, study, recalls, param, param_def=None, patterns=None):
        p = 2 - (param['x'] + 2) ** 2
        eps = 0.0001
        if p < eps:
            p = eps
        n = 1
        return np.log(p), n

    def generate_subject(
        self, study, recall, param, param_def=None, patterns=None, **kwargs
    ):
        recalls_list = [param['recalls']]
        return recalls_list

    def record_subject(self, study, recall, param, param_def=None, patterns=None):
        study_state = study['input']
        recall_state = recall['input']
        return study_state, recall_state


@pytest.fixture()
def data():
    data = fit.sample_data('sample1')
    return data


def test_likelihood_subject(data):
    data = data.copy()
    rec = TestRecall()
    subject_data = data.loc[data['subject'] == 1]
    study, recall = rec.prepare_sim(subject_data)
    param = {'x': -2}
    logl, n = rec.likelihood_subject(study, recall, param)
    np.testing.assert_allclose(logl, np.log(2))


def test_likelihood(data):
    data = data.copy()
    rec = TestRecall()
    param = {'x': -2}
    stats = rec.likelihood(data, param)
    np.testing.assert_allclose(stats['logl'].to_numpy(), [np.log(2), np.log(2)])


def test_fit_subject(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.set_fixed(y=1)
    param_def.set_free(x=[-10, 10])
    subject_data = data.loc[data['subject'] == 1]
    param, logl, n, k = rec.fit_subject(subject_data, param_def)
    np.testing.assert_allclose(param['x'], -2, atol=0.00001)
    np.testing.assert_allclose(logl, np.log(2))
    assert k == 1


def test_fit_indiv(data):
    data = data.copy()
    rec = TestRecall()
    param_def = parameters.Parameters()
    param_def.set_fixed(y=1)
    param_def.set_free(x=[-10, 10])
    results = rec.fit_indiv(data, param_def)
    np.testing.assert_allclose(results['x'].to_numpy(), [-2, -2], atol=0.0001)
    np.testing.assert_allclose(results['logl'].to_numpy(), np.log([2, 2]), atol=0.0001)
    np.testing.assert_array_equal(results['k'].to_numpy(), [1, 1])


def test_generate_subject(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[(data['trial_type'] == 'study') & (data['subject'] == 1)]
    param = {'recalls': ['hollow', 'pupil']}
    # our "model" recalls the positions indicated in the recalls parameter
    rec_list = rec.generate_subject(study, None, param)
    data_sim = fit.add_recalls(study, rec_list)
    expected = ['absence', 'hollow', 'pupil', 'hollow', 'pupil']
    assert data_sim['item'].to_list() == expected


def test_generate(data):
    data = data.copy()
    rec = TestRecall()
    subj_param = {
        1: {'recalls': ['hollow', 'pupil']},
        2: {'recalls': ['pillow', 'fountain', 'piano']},
    }
    sim = rec.generate(data, {}, subj_param)
    expected = [
        'absence',
        'hollow',
        'pupil',
        'hollow',
        'pupil',
        'fountain',
        'piano',
        'pillow',
        'pillow',
        'fountain',
        'piano',
    ]
    assert sim['item'].to_list() == expected


def test_record_subject(data):
    rec = TestRecall()
    study, recall = fit.prepare_lists(data.loc[data['subject'] == 1])
    study_state, recall_state = rec.record_subject(study, recall, {})
    np.testing.assert_array_equal(study_state[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(recall_state[0], np.array([1, 2]))


def test_record(data):
    rec = TestRecall()
    states = rec.record(data, {})
    np.testing.assert_array_equal(
        np.asarray(states), np.array([1, 2, 3, 2, 3, np.nan, 1, 2, 3, 3, 1, 3])
    )
