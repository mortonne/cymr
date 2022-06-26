"""Test management of CMR parameters."""

import numpy as np
import pytest

from cymr import fit
from cymr import cmr


@pytest.fixture()
def param_def_simple():
    param = cmr.CMRParameters()
    param.set_sublayers(f=['task'], c=['task'])
    weights = {(('task', 'item'), ('task', 'item')): 'loc'}
    param.set_weights('fc', weights)
    param.set_weights('cf', weights)
    return param


def test_param_simple(param_def_simple):
    param_def = param_def_simple
    assert param_def.fixed == {}
    assert param_def.free == {}
    assert param_def.dependent == {}
    assert param_def.dynamic == {}
    assert param_def.sublayers == {'f': ['task'], 'c': ['task']}
    assert param_def.weights['fc'] == {(('task', 'item'), ('task', 'item')): 'loc'}
    assert param_def.weights['cf'] == {(('task', 'item'), ('task', 'item')): 'loc'}
    assert param_def.sublayer_param == {}


@pytest.fixture()
def param_def():
    """Parameter definitions."""
    param = cmr.CMRParameters()

    # options
    param.set_options(scope='list')

    # network definition
    param.set_sublayers(f=['task'], c=['loc', 'cat'])
    weights = {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
    }
    param.set_weights('fc', weights)
    param.set_weights('cf', weights)
    param.set_weights('ff', {('task', 'item'): 'loc + cat'})

    # sublayer-varying parameters
    param.set_sublayer_param('c', 'loc', {'B_enc': 'B_enc_loc'})
    param.set_sublayer_param('c', 'cat', {'B_enc': 'B_enc_cat'})
    return param


def test_param(param_def):
    """Test that parameter definitions are correct."""
    assert param_def.options == {'scope': 'list'}
    assert param_def.sublayers == {'f': ['task'], 'c': ['loc', 'cat']}
    assert param_def.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
    }
    assert param_def.weights['cf'] == {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
    }
    assert param_def.weights['ff'] == {('task', 'item'): 'loc + cat'}
    assert param_def.sublayer_param['c'] == {
        'loc': {'B_enc': 'B_enc_loc'},
        'cat': {'B_enc': 'B_enc_cat'},
    }


@pytest.fixture()
def data():
    """Base test DataFrame."""
    data = fit.sample_data('sample1')
    data['list'] = data['subject']
    data['subject'] = 1
    return data


@pytest.fixture()
def patterns():
    cat = np.zeros((24, 3))
    cat[:8, 0] = 1
    cat[8:16, 1] = 1
    cat[16:, 2] = 1
    sim_cat = np.zeros((24, 24))
    sim_cat[:8, :8] = 1
    sim_cat[8:16, 8:16] = 1
    sim_cat[16:, 16:] = 1
    patterns = {
        'vector': {
            'loc': np.eye(24),
            'cat': cat,
        },
        'similarity': {
            'loc': np.eye(24),
            'cat': sim_cat,
        },
    }
    return patterns


def test_eval_weights(param_def, patterns):
    weights = param_def.eval_weights(patterns)
    np.testing.assert_array_equal(
        weights['fc'][(('task', 'item'), ('loc', 'item'))], patterns['vector']['loc']
    )
    np.testing.assert_array_equal(
        weights['fc'][(('task', 'item'), ('cat', 'item'))], patterns['vector']['cat']
    )
    np.testing.assert_array_equal(
        weights['cf'][(('task', 'item'), ('loc', 'item'))], patterns['vector']['loc']
    )
    np.testing.assert_array_equal(
        weights['cf'][(('task', 'item'), ('cat', 'item'))], patterns['vector']['cat']
    )
    np.testing.assert_array_equal(
        weights['ff'][('task', 'item')],
        patterns['similarity']['loc'] + patterns['similarity']['cat'],
    )


def test_eval_weights_index(param_def, patterns):
    item_index = np.arange(12)
    weights = param_def.eval_weights(patterns, item_index=item_index)
    assert weights['fc'][(('task', 'item'), ('loc', 'item'))].shape == (12, 24)
    assert weights['cf'][(('task', 'item'), ('loc', 'item'))].shape == (12, 24)
    assert weights['ff'][('task', 'item')].shape == (12, 12)


def test_eval_sublayer_param(param_def):
    param = {'B_enc_loc': 0.8, 'B_enc_cat': 0.4}
    eval_param = param_def.eval_sublayer_param('c', param, 3)
    expected = np.array([[0.8, 0.4], [0.8, 0.4], [0.8, 0.4]])
    np.testing.assert_array_equal(eval_param['B_enc'], expected)
