"""Models of free recall."""

import numpy as np
from psifr import fr
from cymr.fit import Recall
from cymr import fit
from cymr import network


def prepare_lists(data, study_keys=None, recall_keys=None, clean=True):
    """Prepare study and recall data for simulation."""
    if study_keys is None:
        study_keys = ['input', 'item_index']

    if recall_keys is None:
        recall_keys = ['input']

    s_keys = study_keys.copy()
    s_keys.remove('input')
    r_keys = recall_keys.copy()
    r_keys.remove('input')
    merged = fr.merge_free_recall(data, study_keys=s_keys, recall_keys=r_keys)
    if clean:
        merged = merged.query('~intrusion and repeat == 0')

    study = fr.split_lists(merged, 'study', study_keys)
    recall = fr.split_lists(merged, 'recall', recall_keys)

    for i in range(len(study['input'])):
        if 'input' in study_keys:
            study['input'][i] = study['input'][i].astype(int) - 1
        if 'item_index' in study_keys:
            study['item_index'][i] = study['item_index'][i].astype(int)

        if 'input' in recall_keys:
            recall['input'][i] = recall['input'][i].astype(int) - 1
        if 'item_index' in recall_keys:
            recall['item_index'][i] = recall['item_index'][i].astype(int)

    n = np.unique([len(items) for items in study['input']])
    if len(n) > 1:
        raise ValueError('List length must not vary.')
    return study, recall


def prepare_study(study_data, study_keys=None):
    """Prepare study phase data for simulation."""
    if study_keys is None:
        study_keys = ['position', 'item_index']

    study = fr.split_lists(study_data, 'raw', study_keys)
    for i in range(len(study['position'])):
        if 'position' in study_keys:
            study['position'][i] = study['position'][i].astype(int) - 1
        if 'item_index' in study_keys:
            study['item_index'][i] = study['item_index'][i].astype(int)
    return study


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


def init_dist_cmr(item_index, patterns):
    """Initialize distributed CMR for one list."""
    n_c = patterns['fcf'].shape[1]
    n_f = len(item_index)
    segments = {'item': (n_f, n_c), 'start': (1, 1)}
    net = network.Network(segments)
    list_patterns = patterns['fcf'][item_index]
    net.add_pre_weights('fc', ('item', 'item'), list_patterns)
    net.add_pre_weights('cf', ('item', 'item'), list_patterns)
    net.add_pre_weights('fc', ('start', 'start'), 1)
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
        study, recall = prepare_lists(data, study_keys=['input'],
                                      recall_keys=['input'], clean=True)
        return study, recall

    def likelihood_subject(self, study, recall, param, patterns=None,
                           weights=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        list_param = prepare_list_param(n_item, param)
        net_init = init_loc_cmr(n_item, param)
        logl = 0
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
        return logl

    def generate_subject(self, study_data, param, patterns=None, weights=None,
                         **kwargs):
        study = prepare_study(study_data, study_keys=['position'])
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

    def prepare_sim(self, data):
        study, recall = prepare_lists(data, study_keys=['input', 'item_index'],
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
        for i in range(n_list):
            net = init_dist_cmr(study['item_index'][i], scaled)
            net.study('item', study['input'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            p = net.p_recall('item', recall['input'][i], param['B_rec'],
                             param['T'], list_param['p_stop'])
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
        return logl

    def generate_subject(self, study_data, param, patterns=None, weights=None,
                         **kwargs):
        study = prepare_study(study_data, study_keys=['position', 'item_index'])
        n_item = len(study['position'][0])
        n_list = len(study['position'])
        list_param = prepare_list_param(n_item, param)

        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        recalls = []
        for i in range(n_list):
            net = init_dist_cmr(study['item_index'][i], scaled)
            net.study('item', study['input'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            recall = net.generate_recall('item', param['B_rec'], param['T'],
                                         list_param['p_stop'])
            recalls.append(recall)
        data = fit.add_recalls(study_data, recalls)
        return data
