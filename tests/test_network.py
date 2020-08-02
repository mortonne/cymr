"""Test network operations."""

import os
from itertools import permutations
import pytest
import numpy as np

from cymr import cmr
from cymr import network


@pytest.fixture()
def net():
    segments = {'item': (3, 5), 'task': (1, 1)}
    net = network.Network(segments)
    return net


@pytest.fixture()
def weights():
    mat = np.arange(15).reshape((3, 5))
    return mat


@pytest.fixture()
def net_pre(net, weights):
    net.add_pre_weights('fc', ('item', 'item'), weights)
    net.add_pre_weights('cf', ('item', 'item'), weights)
    net.add_pre_weights('fc', ('task', 'task'), 1)
    return net


@pytest.fixture()
def net_study(net_pre):
    net = net_pre.copy()
    net.update('task', 0)
    B = .5
    L = 1
    for item in range(net.n_f_segment['item']):
        net.present('item', item, B)
        net.learn('fc', 'all', item, L)
        net.learn('cf', 'all', item, L * 2)
    return net


@pytest.fixture()
def net_study_list(net_pre):
    net = net_pre.copy()
    net.update('task', 0)
    B = .5
    Lfc = 1
    Lcf = 2
    item_list = np.arange(net.n_f_segment['item'])
    net.study('item', item_list, B, Lfc, Lcf)
    return net


@pytest.fixture()
def net_study_distract():
    segments = {'item': (2, 5), 'start': (1, 1), 'distract': (3, 3)}
    net = network.Network(segments)
    weights = np.arange(10).reshape((2, 5))
    net.add_pre_weights('fc', ('item', 'item'), weights)
    net.add_pre_weights('cf', ('item', 'item'), weights)
    net.add_pre_weights('fc', ('start', 'start'), 1)
    net.add_pre_weights('fc', ('distract', 'distract'), np.eye(3))
    net.update('start', 0)
    B = .5
    Lfc = 1
    Lcf = 2
    distract_B = .1
    item_list = np.arange(net.n_f_segment['item'])
    distract_list = np.arange(net.n_f_segment['distract'])
    net.study('item', item_list, B, Lfc, Lcf,
              'distract', distract_list, distract_B)
    return net


def test_init_layer():
    layer_segments = {
        'loc': {'item': 3, 'start': 1},
        'cat': {'item': 2, 'start': 1},
        'sem': {'item': 5, 'start': 1}
    }
    layer = network.LayerIndex(layer_segments)

    assert layer.size == 13
    assert layer.size_sublayer['loc'] == 4
    assert layer.size_segment['cat']['item'] == 2

    np.testing.assert_array_equal(layer.get_sublayer('loc'), np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(layer.get_segment('loc', 'item'), np.array([0, 1, 2]))
    assert layer.get_unit('loc', 'item', 1) == 1
    np.testing.assert_array_equal(layer.get_sublayer('cat'), np.array([4, 5, 6]))
    np.testing.assert_array_equal(layer.get_segment('cat', 'start'), np.array([6]))
    assert layer.get_unit('cat', 'item', 1) == 5


def test_network_init(net):
    n_f = net.n_f
    n_c = net.n_c
    assert net.w_cf_exp.shape == (n_f, n_c)
    assert net.w_cf_pre.shape == (n_f, n_c)
    assert net.w_fc_exp.shape == (n_f, n_c)
    assert net.w_fc_pre.shape == (n_f, n_c)
    assert net.c.shape[0] == n_c
    assert net.f.shape[0] == n_f


def test_pre_weights(net_pre, weights):
    net = net_pre
    f_ind, c_ind = net.get_slices(('item', 'item'))
    np.testing.assert_array_equal(net.w_fc_pre[f_ind, c_ind], weights)
    np.testing.assert_array_equal(net.w_cf_pre[f_ind, c_ind], weights)


def test_update(net_pre):
    net = net_pre
    net.update('task', 0)
    expected = np.array([0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(net.c, expected)


def test_present(net_pre):
    net = net_pre
    f_ind, c_ind = net.get_slices(('item', 'item'))
    net.c[0] = 1
    net.present('item', 0, .5)
    np.testing.assert_allclose(np.linalg.norm(net.c, 2), 1)
    expected = np.array(
        [0.8660254, 0.09128709, 0.18257419, 0.27386128, 0.36514837])
    np.testing.assert_allclose(net.c[c_ind], expected)


def test_learn(net_pre):
    net = net_pre
    net.update('task', 0)
    net.present('item', 0, .5)
    net.learn('fc', 'all', 0, 1)
    expected = np.array([0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8660])
    ind = net.get_ind('f', 'item', 0)
    actual = net.w_fc_exp[ind, :]
    np.testing.assert_allclose(actual, expected, atol=.0001)

    net.learn('cf', 'all', 0, 2)
    actual = net.w_cf_exp[ind, :]
    np.testing.assert_allclose(actual, expected * 2, atol=.0001)


def test_study(net_study, net_study_list):
    expected = np.array([[0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8660],
                         [0.1566, 0.2488, 0.3410, 0.4332, 0.5254, 0.5777],
                         [0.2722, 0.3420, 0.4118, 0.4816, 0.5514, 0.3215],
                         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    np.testing.assert_allclose(net_study.w_fc_exp, expected, atol=.0001)
    np.testing.assert_allclose(net_study_list.w_fc_exp, expected, atol=.0001)

    expected = np.array([[0.0000, 0.1826, 0.3651, 0.5477, 0.7303, 1.7321],
                         [0.3131, 0.4975, 0.6819, 0.8663, 1.0507, 1.1553],
                         [0.5444, 0.6840, 0.8236, 0.9632, 1.1029, 0.6429],
                         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    np.testing.assert_allclose(net_study.w_cf_exp, expected, atol=.0001)
    np.testing.assert_allclose(net_study_list.w_cf_exp, expected, atol=.0001)


def test_study_distract(net_study_distract):
    net = net_study_distract
    expected = np.array(
        [[0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8617, 0.0866,
          0.0000, 0.0000],
         [0.1566, 0.2485, 0.3405, 0.4325, 0.5245, 0.5726, 0.0576,
          0.0668, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000]]
    )
    np.testing.assert_allclose(net.w_fc_exp, expected, atol=0.0001)
    expected = np.array(
        [[0.0000, 0.1826, 0.3651, 0.5477, 0.7303, 1.7234, 0.1732,
          0.0000, 0.0000],
         [0.3131, 0.4971, 0.6810, 0.8650, 1.0489, 1.1453, 0.1151,
          0.1336, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000]]
    )
    np.testing.assert_allclose(net.w_cf_exp, expected, atol=0.0001)


def test_recall(net_study):
    net = net_study
    recalls = [2, 0, 1]
    B = .8
    T = 10
    X1 = .05
    X2 = 1
    p_stop = cmr.p_stop_op(len(recalls), X1, X2)
    p = net.p_recall('item', recalls, B, T, p_stop)
    expected = np.array([0.8335545, 0.0760874, 0.6305471, 1.])
    np.testing.assert_allclose(p, expected, atol=.0000001)


def test_sequences(net_study):
    net = net_study
    c_study = net.c.copy()

    # all possible recall sequences
    n_item = 3
    sequences = []
    for i in range(n_item + 1):
        sequences.extend(list(permutations(range(n_item), i)))

    B = .8
    T = 10
    X1 = .05
    X2 = 1
    p_stop = cmr.p_stop_op(n_item, X1, X2)
    p = np.empty((len(sequences), n_item + 1))
    p[:] = np.nan
    for i, recalls in enumerate(sequences):
        net.c = c_study.copy()
        p_recalls = net.p_recall('item', recalls, B, T, p_stop)
        p[i, :len(p_recalls)] = p_recalls

    # probability of any recall sequence should be 1
    p_any = np.sum(np.nanprod(p, 1))
    np.testing.assert_allclose(p_any, 1)


def test_generate(net_study):
    net = net_study.copy()
    B = .8
    T = 10
    X1 = .05
    X2 = 1
    n_item = 3
    p_stop = cmr.p_stop_op(n_item, X1, X2)
    recalls = net.generate_recall('item', B, T, p_stop)


def test_generate_lba(net_study):
    net = net_study.copy()
    B = .8
    time_limit = 90
    A = 4
    b = 8
    s = 1
    tau = 0
    recalls, times = net.generate_recall_lba('item', time_limit, B,
                                             A, b, s, tau)


@pytest.fixture()
def patterns():
    cat = np.array([[1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0]]).T
    patterns = {'vector': {'loc': np.eye(6), 'cat': cat},
                'similarity': {'loc': np.eye(6), 'cat': np.dot(cat, cat.T)}}
    return patterns


def test_pattern_io(patterns):
    temp = 'test_pattern.hdf5'
    items = ['absence', 'hollow', 'pupil', 'fountain', 'piano', 'pillow']
    network.save_patterns(temp, items, loc=patterns['vector']['loc'],
                          cat=patterns['vector']['cat'])
    pat = network.load_patterns(temp)

    # vector representation
    expected = np.array([[1, 0],
                         [0, 1],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [1, 0]])
    np.testing.assert_allclose(pat['vector']['cat'], expected)

    # similarity matrix
    expected = np.array([[1, 0, 1, 1, 0, 1],
                         [0, 1, 0, 0, 1, 0],
                         [1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 1, 0, 1],
                         [0, 1, 0, 0, 1, 0],
                         [1, 0, 1, 1, 0, 1]])
    np.testing.assert_allclose(pat['similarity']['cat'], expected)
    os.remove(temp)


def test_cmr_patterns(patterns):
    weights_template = {'fcf': {'loc': 'w_loc', 'cat': 'w_cat'},
                        'ff': {'loc': 's_loc', 'cat': 's_cat'}}
    params = {'w_loc': 1, 'w_cat': np.sqrt(2), 's_loc': 1, 's_cat': 2}
    weights = network.unpack_weights(weights_template, params)
    scaled = network.prepare_patterns(patterns, weights)
    expected = np.array([[0.57735027, 0., 0., 0., 0., 0., 0.81649658, 0.],
                         [0., 0.57735027, 0., 0., 0., 0., 0., 0.81649658],
                         [0., 0., 0.57735027, 0., 0., 0., 0.81649658, 0.],
                         [0., 0., 0., 0.57735027, 0., 0., 0.81649658, 0.],
                         [0., 0., 0., 0., 0.57735027, 0., 0., 0.81649658],
                         [0., 0., 0., 0., 0., 0.57735027, 0.81649658, 0.]])
    np.testing.assert_allclose(scaled['fcf'], expected)

    expected = np.array([[1., 0., 0.66666667, 0.66666667, 0., 0.66666667],
                         [0., 1., 0., 0., 0.66666667, 0.],
                         [0.66666667, 0., 1., 0.66666667, 0., 0.66666667],
                         [0.66666667, 0., 0.66666667, 1., 0., 0.66666667],
                         [0., 0.66666667, 0., 0., 1., 0.],
                         [0.66666667, 0., 0.66666667, 0.66666667, 0., 1.]])
    np.testing.assert_allclose(scaled['ff'], expected)
