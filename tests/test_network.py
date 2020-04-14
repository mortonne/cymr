"""Test network operations."""

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
    region = ('item', 'item')
    net.add_pre_weights(weights, region)
    net.add_pre_weights(1, ('task', 'task'))
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
