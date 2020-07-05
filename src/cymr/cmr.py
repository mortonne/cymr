"""Models of free recall."""

import numpy as np
from cymr.fit import Recall
from cymr import fit
from cymr import network


def init_loc_cmr(n_item, param):
    """Initialize localist CMR for one list."""
    segments = {'item': (n_item, n_item), 'start': (1, 1)}
    net = network.Network(segments)
    net.add_pre_weights('fc', ('item', 'item'), np.eye(n_item),
                        param['Dfc'], param['Afc'])
    net.add_pre_weights('cf', ('item', 'item'), np.eye(n_item),
                        param['Dcf'], param['Acf'])
    net.add_pre_weights('fc', ('start', 'start'), 1)
    net.update('start', 0)
    return net


def init_dist_cmr(item_index, patterns, param):
    """Initialize distributed CMR for one list."""
    n_c = patterns['fcf'].shape[1]
    n_f = len(item_index)
    segments = {'item': (n_f, n_c), 'start': (1, 1)}
    net = network.Network(segments)
    list_patterns = patterns['fcf'][item_index]
    net.add_pre_weights('fc', ('item', 'item'), list_patterns,
                        param['Dfc'], param['Afc'])
    net.add_pre_weights('cf', ('item', 'item'), list_patterns,
                        param['Dcf'], param['Acf'])
    net.add_pre_weights('fc', ('start', 'start'), 1)
    if 'ff' in patterns and patterns['ff'] is not None:
        mat = patterns['ff'][np.ix_(item_index, item_index)]
        net.add_pre_weights('ff', ('item', 'item'), mat,
                            param['Dff'], param['Aff'])
    net.update('start', 0)
    return net


def prepare_list_param(n_item, param):
    """Prepare parameters that very within list."""
    Lfc = np.tile(param['Lfc'], n_item).astype(float)
    Lcf = network.primacy(n_item, param['Lcf'], param['P1'], param['P2'])
    p_stop = network.p_stop_op(n_item, param['X1'], param['X2'])
    list_param = {'Lfc': Lfc, 'Lcf': Lcf, 'p_stop': p_stop}
    return list_param


class CMR(Recall):

    def prepare_sim(self, data):
        study, recall = fit.prepare_lists(data, study_keys=['input'],
                                          recall_keys=['input'], clean=True)
        return study, recall

    def likelihood_subject(self, study, recall, param, patterns=None,
                           weights=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        list_param = prepare_list_param(n_item, param)
        net_init = init_loc_cmr(n_item, param)
        logl = 0
        n = 0
        for i in range(n_list):
            net = net_init.copy()
            net.study('item', study['input'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            p = net.p_recall('item', recall['input'][i], param['B_rec'],
                             param['T'], list_param['p_stop'])
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
            n += p.size
        return logl, n

    def generate_subject(self, study_data, param, patterns=None, weights=None,
                         **kwargs):
        study = fit.prepare_study(study_data, study_keys=['position'])
        n_item = len(study['position'][0])
        n_list = len(study['position'])
        list_param = prepare_list_param(n_item, param)

        recalls = []
        net_init = init_loc_cmr(n_item, param)
        for i in range(n_list):
            net = net_init.copy()
            net.study('item', study['input'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            recall = net.generate_recall('item', param['B_rec'], param['T'],
                                         list_param['p_stop'])
            recalls.append(recall)
        data = fit.add_recalls(study_data, recalls)
        return data

    def record_network(self, data, param):
        study, recall = self.prepare_sim(data)
        n_item = len(study['input'][0])
        list_param = prepare_list_param(n_item, param)
        net_init = init_loc_cmr(n_item, param)
        n_list = len(study['input'])

        net_state = []
        for i in range(n_list):
            net = net_init.copy()
            item_list = study['input'][i].astype(int)
            state = net.record_study('item', item_list, param['B_enc'],
                                     list_param['Lfc'], list_param['Lcf'])
            rec = net.record_recall('item', recall['input'][i],
                                    param['B_rec'], param['T'])
            state.extend(rec)
            net_state.append(state)
        return net_state


class CMRDistributed(Recall):
    """
    Context Maintenance and Retrieval-Distributed model.

    **Model Parameter Definitions**

    Afc : float
        Intercept of pre-experimental item-context weights.

    Acf : float
        Intercept of pre-experimental context-item weights.

    Aff : float
        Intercept of pre-experimental item-item weights.

    Dfc : float
        Slope of pre-experimental item-context weights.

    Dcf : float
        Slope of pre-experimental context-item weights.

    Dff : float
        Slope of pre-experimental item-item weights.

    Lfc : float
        Learning rate of item-context weights.

    Lcf : float
        Learning rate of context-item weights.

    P1 : float
        Additional context-item learning for first item.

    P2 : float
        Decay rate for primacy learning rate gradient.

    B_enc : float
        Integration rate during encoding.

    B_start : float
        Integration rate of start context reinstatement.

    B_rec : float
        Integration rate during recall.

    X1 : float
        Probability of not recalling any items.

    X2 : float
        Shape parameter of exponential function increasing stop
        probability by output position.

    **Search Parameters**

    All parameters listed above must be defined to evaluate a model.
    In the context of fitting a model using a parameter search,
    parameters may be fixed (i.e., not searched over), free (included
    in the search), or dependent (derived from the other parameters).

    fixed : dict of (str: float)
        Values of fixed parameters.
        Example: :code:`{'Lfc': .8, 'Lcf': .5}`

    free : dict of (str: (float, float))
        Allowed range of free parameters.
        Example: :code:`{'B_enc': (0, 1)}`

    dependent : dict of (str: str), optional
        Expressions to evaluate to set dependent parameters.
        Example: :code:`{'Dfc': '1 - Lfc', 'Dcf': '1 - Lcf'}`

    **Model Patterns**

    Patterns are used to define connections between the item and
    context layers and direct connections between items. Connections
    may be orthonormal as in many published variants of CMR, or they
    may be distributed, overlapping patterns.

    patterns : dict
        May include keys: :code:`'vector'` and/or :code:`'similarity'`.
        Vectors are used to set distributed model representations.
        Similarity matrices are used to set item connections. Vector
        and similarity values are dicts of (feature: array) specifying
        an array for one or more named features, with an
        [items x units] array for vector representations, or
        [items x items] for similarity matrices.
        Example: :code:`{'vector': {'loc': np.eye(24)}}`

    weights : dict
        Keys indicate which model connections to apply weighting
        to. These may include :code:`'fcf'` (applied to
        :math:`M^{FC}_{pre}` and :math:`M^{CF}_{pre}`) and
        :code:`'ff'` (applied to :math:`M^{FF}_{pre}`). Values are
        dicts of (feature: w), where :code:`w` is the name of the
        parameter indicating the scale to apply to a given feature.
        Example: :code:`{'fcf': {'loc': 'w_loc', 'cat': 'w_cat'}}`
    """

    def prepare_sim(self, data):
        study, recall = fit.prepare_lists(
            data, study_keys=['input', 'item_index'],
            recall_keys=['input'], clean=True)
        return study, recall

    def likelihood_subject(self, study, recall, param, patterns=None,
                           weights=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        list_param = prepare_list_param(n_item, param)

        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        logl = 0
        n = 0
        for i in range(n_list):
            net = init_dist_cmr(study['item_index'][i], scaled, param)
            net.study('item', study['input'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            net.integrate('start', 0, param['B_start'])
            p = net.p_recall('item', recall['input'][i], param['B_rec'],
                             param['T'], list_param['p_stop'])
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
            n += p.size
        return logl, n

    def generate_subject(self, study_data, param, patterns=None, weights=None,
                         **kwargs):
        study = fit.prepare_study(study_data,
                                  study_keys=['position', 'item_index'])
        n_item = len(study['position'][0])
        n_list = len(study['position'])
        list_param = prepare_list_param(n_item, param)

        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        recalls = []
        for i in range(n_list):
            net = init_dist_cmr(study['item_index'][i], scaled, param)
            net.study('item', study['position'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            net.integrate('start', 0, param['B_start'])
            recall = net.generate_recall('item', param['B_rec'], param['T'],
                                         list_param['p_stop'])
            recalls.append(recall)
        data = fit.add_recalls(study_data, recalls)
        return data

    def record_network(self, data, param, patterns=None, weights=None,
                       remove_blank=False):
        study, recall = self.prepare_sim(data)
        n_item = len(study['input'][0])
        list_param = prepare_list_param(n_item, param)
        n_list = len(study['input'])

        net_state = []
        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        for i in range(n_list):
            if remove_blank:
                include = np.any(scaled['fcf'][study['item_index'][i]] != 0, 0)
                scaled['fcf'] = scaled['fcf'][:, include]
            net = init_dist_cmr(study['item_index'][i], scaled, param)
            item_list = study['input'][i].astype(int)
            state = net.record_study('item', item_list, param['B_enc'],
                                     list_param['Lfc'], list_param['Lcf'])
            net.integrate('start', 0, param['B_start'])
            rec = net.record_recall('item', recall['input'][i],
                                    param['B_rec'], param['T'])
            state.extend(rec)
            net_state.append(state)
        return net_state
