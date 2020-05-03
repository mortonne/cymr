"""Test operation of models of free recall."""

import numpy as np
import pandas as pd
import pytest
from cymr import models
from cymr import network


@pytest.fixture()
def data():
    data = pd.DataFrame(
        {'subject': [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2],
         'list': [1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2],
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


def test_prepare_lists(data):
    study, recall = models.prepare_lists(data)
    np.testing.assert_array_equal(study['input'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['input'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))
    np.testing.assert_array_equal(recall['input'][0], np.array([1, 2]))
    np.testing.assert_array_equal(recall['input'][1], np.array([2, 0]))


def test_prepare_study(data):
    study_data = data.loc[data['trial_type'] == 'study'].copy()
    study = models.prepare_study(study_data)
    np.testing.assert_array_equal(study['position'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['position'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))


def test_cmr(data):
    model = models.CMR()
    param = {'B_enc': .5, 'B_rec': .8,
             'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    logl = model.likelihood(data, param)
    np.testing.assert_allclose(logl, -5.936799964636842)


def test_cmr_fit(data):
    model = models.CMR()
    fixed = {'B_rec': .8, 'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    var_names = ['B_enc']
    var_bounds = {'B_enc': (0, 1)}
    results = model.fit_indiv(data, fixed, var_names, var_bounds, n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.01)


@pytest.fixture()
def patterns():
    cat = np.array([[1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0]]).T
    patterns = {'vector': {'loc': np.eye(6), 'cat': cat},
                'similarity': {'loc': np.eye(6), 'cat': np.dot(cat, cat.T)}}
    return patterns


def test_init_dist_cmr(patterns):
    weights_template = {'fcf': {'loc': 'w_loc', 'cat': 'w_cat'},
                        'ff': {'loc': 'w_loc', 'cat': 'w_cat'}}
    params = {'w_loc': 1, 'w_cat': 2}
    weights = network.unpack_weights(weights_template, params)
    scaled = network.prepare_patterns(patterns, weights)
    item_index = np.arange(3)
    net = models.init_dist_cmr(item_index, scaled)
    expected = np.array([[0.5774, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.8165, 0.0000, 0.0000],
                         [0.0000, 0.5774, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.8165, 0.0000],
                         [0.0000, 0.0000, 0.5774, 0.0000, 0.0000, 0.0000,
                          0.8165, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.0000, 1.0000]])
    np.testing.assert_allclose(net.w_fc_pre, expected, atol=0.0001)


def test_dist_cmr(data):
    """Test localist CMR using the distributed framework."""
    patterns = {'vector': {'loc': np.eye(6)}}
    weights_template = {'fcf': {'loc': 'w_loc'}}
    param = {'B_enc': .5, 'B_rec': .8, 'w_loc': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}

    model = models.CMR()
    logl = model.likelihood(data, param, patterns=patterns,
                            weights=weights_template)
    np.testing.assert_allclose(logl, -5.936799964636842)


def test_dist_cmr_fit(data):
    model = models.CMR()
    patterns = {'vector': {'loc': np.eye(6)}}
    weights_template = {'fcf': {'loc': 'w_loc'}}
    fixed = {'B_rec': .8, 'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1, 'w_loc': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    var_names = ['B_enc']
    var_bounds = {'B_enc': (0, 1)}
    results = model.fit_indiv(data, fixed, var_names, var_bounds,
                              patterns=patterns, weights=weights_template,
                              n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.01)
