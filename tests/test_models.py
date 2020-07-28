"""Test operation of models of free recall."""

import numpy as np
import pandas as pd
import pytest
from cymr import fit
from cymr import cmr
from cymr import network
from cymr import parameters


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
         'distract': [1, 2, 3, np.nan, np.nan, np.nan,
                      3, 2, 1, np.nan, np.nan, np.nan],
         'task': [1, 2, 1, 2, 1, np.nan,
                  1, 2, 1, 1, 1, 1]})
    return data


def test_prepare_lists(data):
    study, recall = fit.prepare_lists(data)
    np.testing.assert_array_equal(study['input'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['input'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))
    np.testing.assert_array_equal(recall['input'][0], np.array([1, 2]))
    np.testing.assert_array_equal(recall['input'][1], np.array([2, 0]))


def test_prepare_study(data):
    study_data = data.loc[data['trial_type'] == 'study'].copy()
    study = fit.prepare_study(study_data)
    np.testing.assert_array_equal(study['position'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['position'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))


def test_cmr(data):
    model = cmr.CMR()
    param = {'B_enc': .5, 'B_rec': .8,
             'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    logl, n = model.likelihood(data, param)
    np.testing.assert_allclose(logl, -5.936799964636842)
    assert n == 6


@pytest.fixture()
def param_def():
    param_def = parameters.Parameters()
    param_def.set_fixed(
        B_rec=0.8,
        B_start=0,
        Afc=0,
        Dfc=1,
        Acf=0,
        Dcf=1,
        Lfc=1,
        Lcf=1,
        P1=0,
        P2=1,
        T=10,
        X1=0.05,
        X2=1
    )
    return param_def


def test_cmr_fit(data, param_def):
    model = cmr.CMR()
    param_def.set_free(B_enc=(0, 1))
    results = model.fit_indiv(data, param_def, n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.02)
    np.testing.assert_array_equal(results['n'].to_numpy(), [3, 3])


@pytest.fixture()
def patterns():
    cat = np.array([[1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0]]).T
    patterns = {'vector': {'loc': np.eye(6), 'cat': cat},
                'similarity': {'loc': np.eye(6), 'cat': np.dot(cat, cat.T)}}
    return patterns


def test_init_dist_cmr(patterns):
    weights_template = {'fcf': {'loc': 'w_loc', 'cat': 'w_cat'},
                        'ff': {'loc': 's_loc', 'cat': 's_cat'}}
    param = {'w_loc': 1, 'w_cat': np.sqrt(2),
             's_loc': 1, 's_cat': 2,
             'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1, 'Aff': 0, 'Dff': 1}
    weights = network.unpack_weights(weights_template, param)
    scaled = network.prepare_patterns(patterns, weights)
    item_index = np.arange(3)
    net = cmr.init_dist_cmr(item_index, scaled, param)
    expected = np.array([[0.5774, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.8165, 0.0000, 0.0000],
                         [0.0000, 0.5774, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.8165, 0.0000],
                         [0.0000, 0.0000, 0.5774, 0.0000, 0.0000, 0.0000,
                          0.8165, 0.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.0000, 1.0000]])
    np.testing.assert_allclose(net.w_fc_pre, expected, atol=0.0001)

    expected = np.array([[1.0000, 0.0000, 0.6667, 0.0000],
                         [0.0000, 1.0000, 0.0000, 0.0000],
                         [0.6667, 0.0000, 1.0000, 0.0000],
                         [0.0000, 0.0000, 0.0000, 0.0000]])
    np.testing.assert_allclose(net.w_ff_pre, expected, atol=0.0001)


def test_dist_cmr(data):
    """Test localist CMR using the distributed framework."""
    param_def = parameters.Parameters()
    param_def.set_weights('fcf', loc='w_loc')
    patterns = {'vector': {'loc': np.eye(6)}}
    param = {'B_enc': .5, 'B_start': 0, 'B_rec': .8, 'w_loc': 1,
             'Afc': 0, 'Dfc': 1, 'Acf': 1, 'Dcf': 1, 'Aff': 0, 'Dff': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}

    model = cmr.CMRDistributed()
    logl, n = model.likelihood(data, param, None, param_def, patterns=patterns)
    np.testing.assert_allclose(logl, -5.936799964636842)


def test_dist_cmr_fit(data, param_def):
    model = cmr.CMRDistributed()
    patterns = {'vector': {'loc': np.eye(6)}}
    param_def.set_weights('fcf', loc='w_loc')
    param_def.set_fixed(w_loc=1)
    param_def.set_free(B_enc=(0, 1))
    results = model.fit_indiv(data, param_def, patterns=patterns, n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.02)


def test_dynamic_cmr(data):
    patterns = {'vector': {'loc': np.eye(6)}}
    param = {'B_enc': .5, 'B_start': 0, 'B_rec': .8, 'w_loc': 1,
             'Afc': 0, 'Dfc': 1, 'Acf': 1, 'Dcf': 1, 'Aff': 0, 'Dff': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1, 'B_distract': .2}
    param_def = parameters.Parameters()
    param_def.set_dynamic('study', B_enc='distract * B_distract')
    param_def.set_weights('fcf', loc='w_loc')

    model = cmr.CMRDistributed()
    logl, n = model.likelihood(data, param, None, param_def, patterns=patterns,
                               study_keys=['distract'])
    np.testing.assert_allclose(logl, -5.9899248839454415)
