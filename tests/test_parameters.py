"""Test management of parameters."""

import numpy as np
import pandas as pd
import pytest

from psifr import fr
from cymr import parameters


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
def param_def():
    """Parameter definitions."""
    param = parameters.Parameters()
    param.set_fixed(a=1, b=2)
    param.set_fixed({'c': 3})
    param.set_dependent(d='2 + mean([a, b])')
    param.set_dynamic('study', e='distract / c')
    param.set_free(f=[0, 1])
    param.set_weights('fcf', loc='f')
    return param


def test_param(param_def):
    """Test that parameter definitions are correct."""
    assert param_def.fixed == {'a': 1, 'b': 2, 'c': 3}
    assert param_def.free == {'f': [0, 1]}
    assert param_def.dependent == {'d': '2 + mean([a, b])'}
    assert param_def.dynamic == {'study': {'e': 'distract / c'}}
    assert param_def.weights == {'fcf': {'loc': 'f'}}


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
