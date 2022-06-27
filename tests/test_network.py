"""Test network operations."""

from itertools import permutations
import pytest
import numpy as np

from cymr import cmr
from cymr import network


@pytest.fixture()
def net():
    f_segments = {'task': {'item': 3, 'start': 1}}
    c_segments = {'task': {'item': 5, 'start': 1}}
    net = network.Network(f_segments, c_segments)
    return net


@pytest.fixture()
def weights():
    mat = np.arange(15).reshape((3, 5))
    return mat


@pytest.fixture()
def net_pre(net, weights):
    net.add_pre_weights('fc', ('task', 'item'), ('task', 'item'), weights)
    net.add_pre_weights('cf', ('task', 'item'), ('task', 'item'), weights)
    net.add_pre_weights('fc', ('task', 'start'), ('task', 'start'), 1)
    return net


@pytest.fixture()
def net_study(net_pre):
    net = net_pre.copy()
    net.update(('task', 'start', 0), 'task')
    B = 0.5
    L = 1
    n_item = net.f_segment['task']['item']
    for item in range(n_item):
        net.present(('task', 'item', item), 'task', B)
        net.learn('fc', ('task', 'item', item), 'task', L)
        net.learn('cf', ('task', 'item', item), 'task', L * 2)
    return net


@pytest.fixture()
def net_study_list(net_pre):
    net = net_pre.copy()
    net.update(('task', 'start', 0), 'task')
    B = 0.5
    Lfc = 1
    Lcf = 2
    n_item = net.f_segment['task']['item']
    item_list = np.arange(n_item)
    net.study(('task', 'item'), item_list, 'task', B, Lfc, Lcf)
    return net


@pytest.fixture()
def net_study_distract():
    f_segment = {'task': {'item': 2, 'start': 1, 'distract': 3}}
    c_segment = {'task': {'item': 5, 'start': 1, 'distract': 3}}
    net = network.Network(f_segment, c_segment)
    weights = np.arange(10).reshape((2, 5))
    net.add_pre_weights('fc', ('task', 'item'), ('task', 'item'), weights)
    net.add_pre_weights('cf', ('task', 'item'), ('task', 'item'), weights)
    net.add_pre_weights('fc', ('task', 'start'), ('task', 'start'), 1)
    net.add_pre_weights('fc', ('task', 'distract'), ('task', 'distract'), np.eye(3))
    net.update(('task', 'start', 0), 'task')
    B = 0.5
    Lfc = 1
    Lcf = 2
    distract_B = 0.1
    n_item = net.f_segment['task']['item']
    n_distract = net.f_segment['task']['distract']
    item_list = np.arange(n_item)
    distract_list = np.arange(n_distract)
    net.study_distract(
        ('task', 'item'),
        item_list,
        'task',
        B,
        Lfc,
        Lcf,
        ('task', 'distract'),
        distract_list,
        distract_B,
    )
    return net


def test_copy(net_pre):
    net2 = net_pre.copy()
    np.testing.assert_array_equal(net_pre.w_fc_pre, net2.w_fc_pre)

    # test inclusion list
    net3 = net_pre.copy(include=['f', 'c'])
    assert hasattr(net3, 'f') and hasattr(net3, 'c')
    assert not hasattr(net3, 'w_fc_pre')

    # test exclusion list
    net4 = net_pre.copy(exclude=['w_fc_pre'])
    assert hasattr(net4, 'f') and hasattr(net4, 'c')
    assert hasattr(net4, 'w_fc_exp')
    assert not hasattr(net4, 'w_fc_pre')


def test_study_record(net_pre):
    net = net_pre.copy()
    net.update(('task', 'start', 0), 'task')
    B = 0.5
    L = 1
    n_item = net.f_segment['task']['item']
    item_list = np.arange(n_item)
    state = net.record_study(('task', 'item'), item_list, ['task'], B, L, L)

    # presented items
    np.testing.assert_array_equal(state[0].f, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(state[1].f, np.array([0, 1, 0, 0]))
    np.testing.assert_array_equal(state[2].f, np.array([0, 0, 1, 0]))

    # context states
    np.testing.assert_allclose(
        state[0].c,
        np.array([0.0, 0.09128709, 0.18257419, 0.27386128, 0.36514837, 0.8660254]),
    )
    np.testing.assert_allclose(
        state[1].c,
        np.array(
            [0.15655607, 0.24875946, 0.34096284, 0.43316622, 0.52536961, 0.57767384],
        ),
    )
    np.testing.assert_allclose(
        state[2].c,
        np.array(
            [0.27217749, 0.341992, 0.4118065, 0.481621, 0.55143551, 0.32145976],
        ),
    )


def test_recall_record(net_study):
    net = net_study
    recalls = [2, 0, 1]
    B = 0.8
    T = 10
    state = net.record_recall(('task', 'item'), recalls, 'task', B, T)

    # cuing context states
    np.testing.assert_allclose(
        state[0].c,
        np.array(
            [0.27217749, 0.341992, 0.4118065, 0.481621, 0.55143551, 0.32145976],
        ),
    )
    np.testing.assert_allclose(
        state[1].c,
        np.array(
            [0.35085785, 0.39607634, 0.44129483, 0.48651332, 0.53173181, 0.07646726],
        ),
    )
    np.testing.assert_allclose(
        state[2].c,
        np.array(
            [0.07687014, 0.23132715, 0.38578417, 0.54024118, 0.69469819, 0.13146559],
        ),
    )

    # recalled item activation
    np.testing.assert_array_equal(state[0].f, np.array([0, 0, 1, 0]))
    np.testing.assert_array_equal(state[1].f, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(state[2].f, np.array([0, 1, 0, 0]))


@pytest.fixture()
def net_sublayers():
    f_segments = {'task': {'item': 3, 'start': 1}}
    c_segments = {
        'loc': {'item': 3, 'start': 1},
        'cat': {'item': 2, 'start': 1},
        'sem': {'item': 5, 'start': 1},
    }
    net = network.Network(f_segments, c_segments)
    return net


def test_init_layer():
    layer_segments = {
        'loc': {'item': 3, 'start': 1},
        'cat': {'item': 2, 'start': 1},
        'sem': {'item': 5, 'start': 1},
    }
    layer = network.LayerIndex(layer_segments)

    assert layer.size == 13
    assert layer.size_sublayer['loc'] == 4
    assert layer.size_segment['cat']['item'] == 2

    np.testing.assert_array_equal(layer.get_sublayer('loc'), np.array([0, 4]))
    np.testing.assert_array_equal(layer.get_segment('loc', 'item'), np.array([0, 3]))
    assert layer.get_unit('loc', 'item', 1) == 1
    np.testing.assert_array_equal(layer.get_sublayer('cat'), np.array([4, 7]))
    np.testing.assert_array_equal(layer.get_segment('cat', 'start'), np.array([6, 7]))
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


def test_network_copy(net):
    net_copy = net.copy()
    assert net_copy.n_f == net.n_f
    assert net_copy.n_c == net.n_c


def test_get_sublayer(net):
    c_ind = net.get_sublayer('c', 'task')
    np.testing.assert_array_equal(c_ind, np.array([0, 6]))


def test_get_sublayers(net_sublayers):
    c_ind = net_sublayers.get_sublayers('c', ['loc', 'cat', 'sem'])
    np.testing.assert_array_equal(c_ind, np.array([[0, 4], [4, 7], [7, 13]]))


def test_get_region(net):
    f_slice, c_slice = net.get_region(('task', 'item'), ('task', 'start'))
    assert f_slice == slice(0, 3)
    assert c_slice == slice(5, 6)


def test_get_segment(net):
    f_ind = net.get_segment('f', 'task', 'item')
    np.testing.assert_array_equal(f_ind, np.array([0, 3]))


def test_get_slice(net):
    f_slice = net.get_slice('f', 'task', 'item')
    assert f_slice.start == 0
    assert f_slice.stop == 3


def test_get_unit(net):
    ind = net.get_unit('f', 'task', 'start', 0)
    assert ind == 3
    ind = net.get_unit('c', 'task', 'item', 4)
    assert ind == 4


def test_pre_weights(net_pre, weights):
    net = net_pre
    f_slice, c_slice = net.get_region(('task', 'item'), ('task', 'item'))
    np.testing.assert_array_equal(net.w_fc_pre[f_slice, c_slice], weights)
    np.testing.assert_array_equal(net.w_cf_pre[f_slice, c_slice], weights)


def test_update(net_pre):
    net = net_pre
    net.update(('task', 'start', 0), 'task')
    expected = np.array([0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(net.c, expected)


def test_present(net_pre):
    net = net_pre
    net.c[0] = 1
    net.present(('task', 'item', 0), 'task', 0.5)
    np.testing.assert_allclose(np.linalg.norm(net.c, 2), 1)

    c_ind = net.get_segment('c', 'task', 'item')
    expected = np.array([0.8660254, 0.09128709, 0.18257419, 0.27386128, 0.36514837])
    np.testing.assert_allclose(net.c[c_ind[0] : c_ind[1]], expected)


def test_learn(net_pre):
    net = net_pre
    net.update(('task', 'start', 0), 'task')
    net.present(('task', 'item', 0), 'task', 0.5)
    net.learn('fc', ('task', 'item', 0), 'task', 1)

    expected = np.array([0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8660])
    ind = net.get_unit('f', 'task', 'item', 0)
    actual = net.w_fc_exp[ind, :]
    np.testing.assert_allclose(actual, expected, atol=0.0001)

    net.learn('cf', ('task', 'item', 0), 'task', 2)
    actual = net.w_cf_exp[ind, :]
    np.testing.assert_allclose(actual, expected * 2, atol=0.0001)


def test_study(net_study, net_study_list):
    expected = np.array(
        [
            [0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8660],
            [0.1566, 0.2488, 0.3410, 0.4332, 0.5254, 0.5777],
            [0.2722, 0.3420, 0.4118, 0.4816, 0.5514, 0.3215],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    np.testing.assert_allclose(net_study.w_fc_exp, expected, atol=0.0001)
    np.testing.assert_allclose(net_study_list.w_fc_exp, expected, atol=0.0001)

    expected = np.array(
        [
            [0.0000, 0.1826, 0.3651, 0.5477, 0.7303, 1.7321],
            [0.3131, 0.4975, 0.6819, 0.8663, 1.0507, 1.1553],
            [0.5444, 0.6840, 0.8236, 0.9632, 1.1029, 0.6429],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    np.testing.assert_allclose(net_study.w_cf_exp, expected, atol=0.0001)
    np.testing.assert_allclose(net_study_list.w_cf_exp, expected, atol=0.0001)


def test_study_distract(net_study_distract):
    net = net_study_distract
    expected = np.array(
        [
            [0.0000, 0.0913, 0.1826, 0.2739, 0.3651, 0.8617, 0.0866, 0.0000, 0.0000],
            [0.1566, 0.2485, 0.3405, 0.4325, 0.5245, 0.5726, 0.0576, 0.0668, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    np.testing.assert_allclose(net.w_fc_exp, expected, atol=0.0001)
    expected = np.array(
        [
            [0.0000, 0.1826, 0.3651, 0.5477, 0.7303, 1.7234, 0.1732, 0.0000, 0.0000],
            [0.3131, 0.4971, 0.6810, 0.8650, 1.0489, 1.1453, 0.1151, 0.1336, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    np.testing.assert_allclose(net.w_cf_exp, expected, atol=0.0001)


def test_recall(net_study):
    net = net_study
    recalls = [2, 0, 1]
    B = 0.8
    T = 10
    X1 = 0.05
    X2 = 1
    p_stop = cmr.p_stop_op(len(recalls), X1, X2)
    p = net.p_recall(('task', 'item'), recalls, 'task', B, T, p_stop)
    expected = np.array([0.8335545, 0.0760874, 0.6305471, 1.0])
    np.testing.assert_allclose(p, expected, atol=0.0000001)


def test_sequences(net_study):
    net = net_study
    c_study = net.c.copy()

    # all possible recall sequences
    n_item = 3
    sequences = []
    for i in range(n_item + 1):
        sequences.extend(list(permutations(range(n_item), i)))

    B = 0.8
    T = 10
    X1 = 0.05
    X2 = 1
    p_stop = cmr.p_stop_op(n_item, X1, X2)
    p = np.empty((len(sequences), n_item + 1))
    p[:] = np.nan
    for i, recalls in enumerate(sequences):
        net.c = c_study.copy()
        p_recalls = net.p_recall(('task', 'item'), recalls, 'task', B, T, p_stop)
        p[i, : len(p_recalls)] = p_recalls

    # probability of any recall sequence should be 1
    p_any = np.sum(np.nanprod(p, 1))
    np.testing.assert_allclose(p_any, 1)


def test_generate(net_study):
    net = net_study.copy()
    B = 0.8
    T = 10
    X1 = 0.05
    X2 = 1
    n_item = 3
    p_stop = cmr.p_stop_op(n_item, X1, X2)
    recalls = net.generate_recall(('task', 'item'), 'task', B, T, p_stop)


def test_generate_lba(net_study):
    net = net_study.copy()
    B = 0.8
    time_limit = 90
    A = 4
    b = 8
    s = 1
    tau = 0
    recalls, times = net.generate_recall_lba(
        ('task', 'item'), 'task', time_limit, B, A, b, s, tau
    )


@pytest.fixture()
def patterns():
    cat = np.array([[1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]]).T
    patterns = {
        'vector': {'loc': np.eye(6), 'cat': cat},
        'similarity': {'loc': np.eye(6), 'cat': np.dot(cat, cat.T)},
    }
    return patterns


def test_cmr_patterns(patterns):
    param_def = cmr.CMRParameters()
    fcf_weights = {
        (('task', 'item'), ('task', 'loc')): 'w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'w_cat * cat',
    }
    ff_weights = {('task', 'item'): 's_loc * loc + s_cat * cat'}
    param_def.set_weights('fc', fcf_weights)
    param_def.set_weights('ff', ff_weights)
    param_def.set_dependent(
        {
            'w_loc': 'wr_loc / sqrt(wr_loc**2 + wr_cat**2)',
            'w_cat': 'wr_cat / sqrt(wr_loc**2 + wr_cat**2)',
            's_loc': 'sr_loc / (sr_loc + sr_cat)',
            's_cat': 'sr_cat / (sr_loc + sr_cat)',
        }
    )
    param = {'wr_loc': 1, 'wr_cat': np.sqrt(2), 'sr_loc': 1, 'sr_cat': 2}
    param = param_def.eval_dependent(param)
    weights = param_def.eval_weights(patterns, param)

    # localist FC units
    expected = np.array(
        [
            [0.57735027, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.57735027, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.57735027, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.57735027, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.57735027, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.57735027],
        ]
    )
    np.testing.assert_allclose(
        weights['fc'][(('task', 'item'), ('task', 'loc'))], expected
    )

    # category FC units
    expected = np.array(
        [
            [0.81649658, 0.0],
            [0.0, 0.81649658],
            [0.81649658, 0.0],
            [0.81649658, 0.0],
            [0.0, 0.81649658],
            [0.81649658, 0.0],
        ]
    )
    np.testing.assert_allclose(
        weights['fc'][(('task', 'item'), ('task', 'cat'))], expected
    )

    # FF units
    expected = np.array(
        [
            [1.0, 0.0, 0.66666667, 0.66666667, 0.0, 0.66666667],
            [0.0, 1.0, 0.0, 0.0, 0.66666667, 0.0],
            [0.66666667, 0.0, 1.0, 0.66666667, 0.0, 0.66666667],
            [0.66666667, 0.0, 0.66666667, 1.0, 0.0, 0.66666667],
            [0.0, 0.66666667, 0.0, 0.0, 1.0, 0.0],
            [0.66666667, 0.0, 0.66666667, 0.66666667, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(weights['ff'][('task', 'item')], expected)
