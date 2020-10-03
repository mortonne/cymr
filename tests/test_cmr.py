"""Test operation of models of free recall."""

import numpy as np
import pandas as pd
import pytest
from cymr import fit
from cymr import cmr
from cymr import parameters


@pytest.fixture()
def data():
    """Generate sample free recall data."""
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
         'op': [1, 2, 3, 1, 2, 3,
                1, 2, 3, 1, 2, 3],
         'task': [1, 2, 1, 2, 1, np.nan,
                  1, 2, 1, 1, 1, 1]})
    return data


def test_prepare_lists(data):
    """Test splitting lists into study and recall list format."""
    study, recall = fit.prepare_lists(data)
    np.testing.assert_array_equal(study['input'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['input'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))
    np.testing.assert_array_equal(recall['input'][0], np.array([1, 2]))
    np.testing.assert_array_equal(recall['input'][1], np.array([2, 0]))


def test_prepare_study(data):
    """Test splitting study lists."""
    study_data = data.loc[data['trial_type'] == 'study'].copy()
    study = fit.prepare_study(study_data)
    np.testing.assert_array_equal(study['position'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['position'][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(study['item_index'][1], np.array([3, 4, 5]))


def test_cmr(data):
    """Test CMR likelihood evaluation."""
    model = cmr.CMRDistributed()
    param = {'B_enc': .5, 'B_rec': .8, 'B_start': 0,
             'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    param_def, patterns = cmr.config_loc_cmr(768)
    logl, n = model.likelihood(data, param, param_def=param_def, patterns=patterns)
    np.testing.assert_allclose(logl, -5.936799964636842)
    assert n == 6


@pytest.fixture()
def param_def():
    """Generate a parameter definition with standard fixed values."""
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
    """Test fit of CMR parameters to sample data."""
    model = cmr.CMR()
    param_def.set_free(B_enc=(0, 1))
    results = model.fit_indiv(data, param_def, n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.02)
    np.testing.assert_array_equal(results['n'].to_numpy(), [3, 3])


@pytest.fixture()
def patterns():
    """Generate patterns for use in CMR-D."""
    cat = np.array([[1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0]]).T
    items = np.array(['absence', 'hollow', 'pupil', 'fountain', 'piano', 'pillow'])
    patterns = {
        'items': items,
        'vector': {'loc': np.eye(6), 'cat': cat},
        'similarity': {'loc': np.eye(6), 'cat': np.dot(cat, cat.T)},
    }
    return patterns


def test_init_network(patterns):
    """Test initialization of a complex network."""
    param_def = parameters.Parameters()
    param_def.set_dependent(
        w_loc='wr_loc / sqrt(wr_loc**2 + wr_cat**2)',
        w_cat='wr_cat / sqrt(wr_loc**2 + wr_cat**2)',
        s_loc='sr_loc / (sr_loc + sr_cat)',
        s_cat='sr_cat / (sr_loc + sr_cat)',
    )
    param_def.set_sublayers(f=['task'], c=['loc', 'cat'])
    param_def.set_weights('fc', {
        (('task', 'item'), ('loc', 'item')): 'w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'w_cat * cat',
    })
    param_def.set_weights('cf', {
        (('task', 'item'), ('loc', 'item')): 'w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'w_cat * cat',
    })
    param_def.set_weights('ff', {
        ('task', 'item'): 's_loc * loc + s_cat * cat'
    })
    item_index = np.arange(3)
    param = {'wr_loc': 1, 'wr_cat': np.sqrt(2), 'sr_loc': 1, 'sr_cat': 2}
    param = param_def.eval_dependent(param)
    net = cmr.init_network(param_def, patterns, param, item_index)

    expected = np.array(
        [[0.5774, 0.0000, 0.0000, 0, 0, 0, 0.0000, 0.8165, 0.0000, 0.0000],
         [0.0000, 0.5774, 0.0000, 0, 0, 0, 0.0000, 0.0000, 0.8165, 0.0000],
         [0.0000, 0.0000, 0.5774, 0, 0, 0, 0.0000, 0.8165, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0, 0, 0, 1.0000, 0.0000, 0.0000, 1.0000]]
    )
    np.testing.assert_allclose(net.w_fc_pre, expected, atol=0.0001)

    expected = np.array(
        [[1.0000, 0.0000, 0.6667, 0.0000],
         [0.0000, 1.0000, 0.0000, 0.0000],
         [0.6667, 0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]]
    )
    np.testing.assert_allclose(net.w_ff_pre, expected, atol=0.0001)


@pytest.fixture()
def param_def_dist(param_def):
    """Generate parameter definitions for a simple CMR-D network."""
    param_def = param_def.copy()
    param_def.set_sublayers(f=['task'], c=['task'])
    weights = {(('task', 'item'), ('task', 'item')): 'loc'}
    param_def.set_weights('fc', weights)
    param_def.set_weights('cf', weights)
    return param_def


@pytest.fixture()
def param_dist():
    """Generate standard parameters for testing CMR-D."""
    param = {
        'B_enc': .5, 'B_start': 0, 'B_rec': .8,
        'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
        'T': 10, 'X1': .05, 'X2': 1
    }
    return param


def test_dist_cmr(data, patterns, param_def_dist, param_dist):
    """Test localist CMR using the CMR-D implementation."""
    model = cmr.CMRDistributed()
    logl, n = model.likelihood(
        data, param_dist, None, param_def_dist, patterns=patterns
    )
    np.testing.assert_allclose(logl, -5.936799964636842)


def test_dist_cmr_fit(data, patterns, param_def_dist):
    """Test fitted parameter values for CMR-D."""
    param_def = param_def_dist.copy()
    model = cmr.CMRDistributed()
    param_def.set_fixed(w_loc=1)
    param_def.set_free(B_enc=(0, 1))
    results = model.fit_indiv(data, param_def, patterns=patterns, n_jobs=2)
    np.testing.assert_allclose(results['B_enc'].to_numpy(),
                               np.array([0.72728744, 0.99883425]), atol=0.02)


def test_dist_cmr_generate(data, patterns, param_def_dist, param_dist):
    """Test that CMR-D generation runs."""
    model = cmr.CMRDistributed()
    sim = model.generate(data, param_dist, None, param_def_dist, patterns=patterns)
    assert isinstance(sim, pd.DataFrame)


def test_dynamic_cmr(data, patterns, param_def_dist, param_dist):
    """Test evaluation of a dynamic study parameter."""
    param = param_dist.copy()
    param_def = param_def_dist.copy()
    param['B_distract'] = .2
    param_def.set_dynamic('study', B_enc='distract * B_distract')
    model = cmr.CMRDistributed()
    logl, n = model.likelihood(data, param, None, param_def, patterns=patterns,
                               study_keys=['distract'])
    np.testing.assert_allclose(logl, -5.9899248839454415)


def test_dynamic_cmr_recall(data, patterns, param_def_dist, param_dist):
    """Test evaluation of a dynamic recall parameter."""
    param = param_dist.copy()
    param_def = param_def_dist.copy()
    param['B_op'] = .2
    param_def.set_dynamic('recall', B_rec='op * B_op')
    model = cmr.CMRDistributed()
    logl, n = model.likelihood(data, param, None, param_def, patterns=patterns,
                               recall_keys=['op'])
    np.testing.assert_allclose(logl, -5.919470385031945)


@pytest.fixture()
def param_def_sublayer():
    """Generate parameter definitions for multiple context sublayers."""
    param_def = parameters.Parameters()
    param_def.set_sublayers(f=['task'], c=['loc', 'cat'])
    param_def.set_sublayer_param('c', 'loc', {'B_enc': 'B_enc_loc'})
    param_def.set_sublayer_param('c', 'cat', {'B_enc': 'B_enc_cat'})
    weights = {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
    }
    param_def.set_weights('fc', weights)
    param_def.set_weights('cf', weights)
    return param_def


def test_sublayer_study(data, patterns, param_def_sublayer, param_dist):
    """Test steps involved in working with context sublayers."""
    # test expanded list parameters
    param = param_dist.copy()
    param['B_enc_loc'] = .5
    param['B_enc_cat'] = .8
    n_item = 3
    n_sub = 2
    list_param = cmr.prepare_list_param(
        n_item, n_sub, param, param_def_sublayer
    )
    np.testing.assert_array_equal(
        list_param['Lfc'], np.array([[1, 1], [1, 1], [1, 1]])
    )
    np.testing.assert_array_equal(
        list_param['Lcf'], np.array([[1, 1], [1, 1], [1, 1]])
    )

    # test sublayer-specific parameters
    assert param_def_sublayer.sublayers['c'] == ['loc', 'cat']
    list_param = param_def_sublayer.eval_sublayer_param(
        'c', list_param, n_item
    )
    np.testing.assert_array_equal(
        list_param['B_enc'], np.array([[.5, .8], [.5, .8], [.5, .8]])
    )

    # prepare lists for simulation
    study, recall = fit.prepare_lists(
        data, study_keys=['input', 'item_index'], recall_keys=['input'], clean=True
    )

    # study the first list
    net = cmr.study_list(
        param_def_sublayer, list_param, study['item_index'][0],
        study['input'][0], patterns
    )
    expected = np.array(
        [[0.5, 0., 0., 0., 0., 0., 0.8660, 0.8, 0., 0.6],
         [0.4330, 0.5, 0., 0., 0., 0., 0.75, 0.48, 0.8, 0.36],
         [0.375, 0.4330, 0.5, 0., 0., 0., 0.6495, 0.9576, 0.2627, 0.1182],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    np.testing.assert_allclose(net.w_fc_exp, expected, atol=0.0001)


def test_sublayer_cmr(data, patterns, param_def_sublayer, param_dist):
    """Test evaluation of CMR-D with multiple context sublayers."""
    param = param_dist.copy()
    param['B_enc_loc'] = .5
    param['B_enc_cat'] = .8
    model = cmr.CMRDistributed()
    logl, n = model.likelihood(
        data, param, None, param_def_sublayer, patterns=patterns
    )
    np.testing.assert_allclose(logl, -5.8694368046085215)


def test_sublayer_generate(data, patterns, param_def_sublayer, param_dist):
    """Test using CMR-D with sublayers to generate data."""
    param = param_dist.copy()
    param['B_enc_loc'] = .5
    param['B_enc_cat'] = .8
    model = cmr.CMRDistributed()
    sim = model.generate(
        data, param, None, param_def_sublayer, patterns=patterns
    )
    assert isinstance(sim, pd.DataFrame)
