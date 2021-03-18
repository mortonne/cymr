"""Test management of parameters."""

import numpy as np
import pandas as pd
import pytest

from psifr import fr
from cymr import parameters


@pytest.fixture()
def param_def_simple():
    param = parameters.Parameters()
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
    assert param_def.weights['fc'] == {
        (('task', 'item'), ('task', 'item')): 'loc'
    }
    assert param_def.weights['cf'] == {
        (('task', 'item'), ('task', 'item')): 'loc'
    }
    assert param_def.sublayer_param == {}


@pytest.fixture()
def param_def():
    """Parameter definitions."""
    param = parameters.Parameters()

    # options
    param.set_options(scope='list')

    # general parameter management
    param.set_fixed(a=1, b=2)
    param.set_fixed({'c': 3})
    param.set_dependent(d='2 + mean([a, b])')
    param.set_dynamic('study', e='distract / c')
    param.set_free(f=[0, 1])

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
    assert param_def.fixed == {'a': 1, 'b': 2, 'c': 3}
    assert param_def.free == {'f': [0, 1]}
    assert param_def.dependent == {'d': '2 + mean([a, b])'}
    assert param_def.dynamic == {'study': {'e': 'distract / c'}}
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
    data = pd.DataFrame({
        'subject': [1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1],
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
                 1, 2, 1, 1, 1, 1],
        'distract': [1, 2, 3, np.nan, np.nan, np.nan,
                     3, 2, 1, np.nan, np.nan, np.nan],
    })
    return data


@pytest.fixture()
def split_data(data):
    """Data split into study and recall."""
    merged = fr.merge_free_recall(data, study_keys=['distract'])
    split = {
        'study': fr.split_lists(merged, 'study', ['input', 'distract']),
        'recall': fr.split_lists(merged, 'recall', ['input']),
    }
    return split


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
        }
    }
    return patterns


def test_set_dependent():
    param = {'Lfc': .7}
    dependent = {'Dfc': '1 - Lfc'}
    updated = parameters.set_dependent(param, dependent)
    expected = {'Lfc': .7, 'Dfc': .3}
    np.testing.assert_allclose(updated['Dfc'], expected['Dfc'])


def test_set_dynamic(data):
    param = {'B_distract': .2}
    dynamic = {'study': {'B_enc': 'distract * B_distract'}}
    study_data = fr.filter_data(data, 1, 1, 'study')
    study = fr.split_lists(study_data, 'raw', ['distract'])
    updated = parameters.set_dynamic(param, study, dynamic['study'])
    expected = {'B_distract': .2, 'B_enc': [np.array([.2, .4, .6])]}
    np.testing.assert_allclose(updated['B_enc'][0], expected['B_enc'][0])


def test_dependent(param_def):
    """Test evaluation of dependent parameters."""
    param = {'a': 1, 'b': 2}
    param = param_def.eval_dependent(param)
    assert param == {'a': 1, 'b': 2, 'd': 3.5}


def test_dynamic(param_def, split_data):
    """Test evaluation of dynamic parameters."""
    param = {'c': 2}
    param = param_def.eval_dynamic(param, study=split_data['study'])
    np.testing.assert_array_equal(param['e'][0], np.array([0.5, 1, 1.5]))
    np.testing.assert_array_equal(param['e'][1], np.array([1.5, 1, 0.5]))


def test_get_dynamic(param_def, split_data):
    """Test indexing of dynamic parameters."""
    param = {'c': 2}
    param = param_def.eval_dynamic(param, study=split_data['study'])

    param1 = param_def.get_dynamic(param, 0)
    np.testing.assert_array_equal(param1['e'], np.array([0.5, 1, 1.5]))
    param2 = param_def.get_dynamic(param, 1)
    np.testing.assert_array_equal(param2['e'], np.array([1.5, 1, 0.5]))


def test_blank(split_data):
    """Test that unspecified evaluation does nothing."""
    param_def = parameters.Parameters()
    param = {'a': 1, 'b': 2, 'c': 3}
    orig = param.copy()
    param = param_def.eval_dependent(param)
    param = param_def.eval_dynamic(
        param, study=split_data['study'], recall=split_data['recall']
    )
    assert param == orig


def test_eval_weights(param_def, patterns):
    weights = param_def.eval_weights(patterns)
    np.testing.assert_array_equal(
        weights['fc'][(('task', 'item'), ('loc', 'item'))],
        patterns['vector']['loc']
    )
    np.testing.assert_array_equal(
        weights['fc'][(('task', 'item'), ('cat', 'item'))],
        patterns['vector']['cat']
    )
    np.testing.assert_array_equal(
        weights['cf'][(('task', 'item'), ('loc', 'item'))],
        patterns['vector']['loc']
    )
    np.testing.assert_array_equal(
        weights['cf'][(('task', 'item'), ('cat', 'item'))],
        patterns['vector']['cat']
    )
    np.testing.assert_array_equal(
        weights['ff'][('task', 'item')],
        patterns['similarity']['loc'] + patterns['similarity']['cat']
    )


def test_eval_weights_index(param_def, patterns):
    item_index = np.arange(12)
    weights = param_def.eval_weights(patterns, item_index=item_index)
    assert weights['fc'][(('task', 'item'), ('loc', 'item'))].shape == (12, 24)
    assert weights['cf'][(('task', 'item'), ('loc', 'item'))].shape == (12, 24)
    assert weights['ff'][('task', 'item')].shape == (12, 12)


def test_eval_sublayer_param(param_def):
    param = {'B_enc_loc': .8, 'B_enc_cat': .4}
    eval_param = param_def.eval_sublayer_param('c', param, 3)
    expected = np.array(
        [[.8, .4],
         [.8, .4],
         [.8, .4]]
    )
    np.testing.assert_array_equal(eval_param['B_enc'], expected)


def test_json(param_def, tmp_path):
    p = tmp_path / 'parameters.json'
    param_def.to_json(p.as_posix())
    param = parameters.read_json(p.as_posix())
    assert param.fixed == param_def.fixed
    assert param.free == param_def.free
    assert param.dependent == param_def.dependent
    assert param.dynamic == param_def.dynamic
    assert param.weights == param_def.weights
