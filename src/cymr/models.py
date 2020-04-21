"""Models of free recall."""

import numpy as np
from psifr import fr
from cymr.fit import Recall
from cymr import fit
from cymr import network


class CMR(Recall):

    def prepare_sim(self, data):
        # prepare list data for simulation
        data_study = data.loc[data['trial_type'] == 'study']
        data_recall = data.loc[data['trial_type'] == 'recall']
        merged = fr.merge_lists(data_study, data_recall)
        merged = merged.query('~intrusion and repeat == 0')

        study = fr.split_lists(merged, 'study', ['input'])
        recall = fr.split_lists(merged, 'recall', ['input'])
        for i in range(len(study['input'])):
            study['input'][i] -= 1
            recall['input'][i] -= 1
            study['input'][i] = study['input'][i].astype(int)
            recall['input'][i] = recall['input'][i].astype(int)
        n = np.unique([len(items) for items in study['input']])
        if len(n) > 1:
            raise ValueError('List length must not vary.')
        return study, recall

    def init_network(self, n_item):
        segments = {'item': (n_item, n_item), 'start': (1, 1)}
        net = network.Network(segments)
        net.add_pre_weights('fc', ('item', 'item'), np.eye(n_item))
        net.add_pre_weights('cf', ('item', 'item'), np.eye(n_item))
        net.add_pre_weights('fc', ('start', 'start'), 1)
        net.update('start', 0)
        return net

    def likelihood_subject(self, study, recall, param):
        n_item = len(study['input'][0])
        net_init = self.init_network(n_item)
        n_list = len(study['input'])
        p_stop = network.p_stop_op(n_item, param['X1'], param['X2'])
        logl = 0
        for i in range(n_list):
            net = net_init.copy()
            item_list = study['input'][i]
            net.study('item', item_list, param['B_enc'], param['L'], param['L'])
            p = net.p_recall('item', recall['input'][i], param['B_rec'],
                             param['T'], p_stop)
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
        return logl

    def generate_subject(self, study_data, param, **kwargs):
        study = fr.split_lists(study_data, 'raw', ['position'])
        n_item = len(study['position'][0])
        n_list = len(study['position'])
        net_init = self.init_network(n_item)
        p_stop = network.p_stop_op(n_item, param['X1'], param['X2'])
        recalls = []
        for i in range(n_list):
            net = net_init.copy()
            item_list = study['position'][i].astype(int)
            net.study('item', item_list - 1, param['B_enc'], param['L'],
                      param['L'])
            recall = net.generate_recall('item', param['B_rec'], param['T'],
                                         p_stop)
            recalls.append(recall)
        data = fit.add_recalls(study_data, recalls)
        return data
