"""Test operation of models of free recall."""

import numpy as np
import pandas as pd
import pytest
from cymr import models


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


@pytest.fixture()
def patterns():
    cat = np.array([[1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0]])
    patterns = {'fcf': {'loc': np.eye(6), 'cat': cat.T}}
    return patterns


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


def test_cmr_patterns(patterns):
    weights = {'fcf': {'loc': 1, 'cat': 2}}
    scaled = models.prepare_patterns(patterns, weights)
    expected = np.array([[0.57735027, 0., 0., 0., 0., 0., 0.81649658, 0.],
                         [0., 0.57735027, 0., 0., 0., 0., 0., 0.81649658],
                         [0., 0., 0.57735027, 0., 0., 0., 0.81649658, 0.],
                         [0., 0., 0., 0.57735027, 0., 0., 0.81649658, 0.],
                         [0., 0., 0., 0., 0.57735027, 0., 0., 0.81649658],
                         [0., 0., 0., 0., 0., 0.57735027, 0.81649658, 0.]])
    np.testing.assert_allclose(scaled['fcf'], expected)
