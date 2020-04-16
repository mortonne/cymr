"""Test network operations."""

from itertools import permutations
import pytest
import numpy as np
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
    net.add_pre_weights(weights, ('item', 'item'))
    net.add_pre_weights(1, ('task', 'task'))
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
    net.add_pre_weights(weights, ('item', 'item'))
    net.add_pre_weights(1, ('start', 'start'))
    net.add_pre_weights(np.eye(3), ('distract', 'distract'))
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
    expected = np.array([[0.0000,  0.0913,  0.1826,  0.2739,  0.3651,  0.8660],
                         [0.1566,  0.2488,  0.3410,  0.4332,  0.5254,  0.5777],
                         [0.2722,  0.3420,  0.4118,  0.4816,  0.5514,  0.3215],
                         [0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
    np.testing.assert_allclose(net_study.w_fc_exp, expected, atol=.0001)
    np.testing.assert_allclose(net_study_list.w_fc_exp, expected, atol=.0001)

    expected = np.array([[0.0000,  0.1826,  0.3651,  0.5477,  0.7303,  1.7321],
                         [0.3131,  0.4975,  0.6819,  0.8663,  1.0507,  1.1553],
                         [0.5444,  0.6840,  0.8236,  0.9632,  1.1029,  0.6429],
                         [0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
    np.testing.assert_allclose(net_study.w_cf_exp, expected, atol=.0001)
    np.testing.assert_allclose(net_study_list.w_cf_exp, expected, atol=.0001)


def test_recall(net_study):
    net = net_study
    recalls = [2, 0, 1]
    B = .8
    T = 10
    X1 = .05
    X2 = 1
    p_stop = network.p_stop_op(len(recalls), X1, X2)
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
    p_stop = network.p_stop_op(n_item, X1, X2)
    p = np.empty((len(sequences), n_item + 1))
    p[:] = np.nan
    for i, recalls in enumerate(sequences):
        net.c = c_study.copy()
        p_recalls = net.p_recall('item', recalls, B, T, p_stop)
        p[i, :len(p_recalls)] = p_recalls

    # probability of any recall sequence should be 1
    p_any = np.sum(np.nanprod(p, 1))
    np.testing.assert_allclose(p_any, 1)
