"""Test management of parameters."""

import numpy as np
import pytest

from psifr import fr
from cymr import fit
from cymr import parameters


@pytest.fixture()
def param_def():
    """Parameter definitions."""
    param = parameters.Parameters()

    # general parameter management
    param.set_fixed(a=1, b=2)
    param.set_fixed({'c': 3})
    param.set_dependent(d='2 + mean([a, b])')
    param.set_dynamic('study', e='distract / c')
    param.set_free(f=[0, 1])
    return param


def test_param(param_def):
    """Test that parameter definitions are correct."""
    assert param_def.fixed == {'a': 1, 'b': 2, 'c': 3}
    assert param_def.free == {'f': [0, 1]}
    assert param_def.dependent == {'d': '2 + mean([a, b])'}
    assert param_def.dynamic == {'study': {'e': 'distract / c'}}


@pytest.fixture()
def data():
    """Base test DataFrame."""
    data = fit.sample_data('sample1')
    data['list'] = data['subject']
    data['subject'] = 1
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


def test_set_dependent():
    param = {'Lfc': 0.7}
    dependent = {'Dfc': '1 - Lfc'}
    updated = parameters.set_dependent(param, dependent)
    expected = {'Lfc': 0.7, 'Dfc': 0.3}
    np.testing.assert_allclose(updated['Dfc'], expected['Dfc'])


def test_set_dynamic(data):
    param = {'B_distract': 0.2}
    dynamic = {'study': {'B_enc': 'distract * B_distract'}}
    study_data = fr.filter_data(data, 1, 1, 'study')
    study = fr.split_lists(study_data, 'raw', ['distract'])
    updated = parameters.set_dynamic(param, study, dynamic['study'])
    expected = {'B_distract': 0.2, 'B_enc': [np.array([0.2, 0.4, 0.6])]}
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


def test_json(param_def, tmp_path):
    p = tmp_path / 'parameters.json'
    param_def.to_json(p.as_posix())
    param = parameters.read_json(p.as_posix())
    assert param.options == param_def.options
    assert param.fixed == param_def.fixed
    assert param.free == param_def.free
    assert param.dependent == param_def.dependent
    assert param.dynamic == param_def.dynamic
