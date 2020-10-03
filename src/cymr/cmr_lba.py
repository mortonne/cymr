
import numpy as np
import pandas as pd

from cymr import operations
from cymr import fit
from cymr import network
from cymr import cmr
from cymr.cmr import CMR
from cymr import lba


def add_recalls(study, recalls_list, recalls_times):
    """Doc"""
    lists = study['list'].unique()
    subjects = study['subject'].unique()
    if len(subjects) > 1:
        raise ValueError('Unpacking multiple subjects not supported.')
    subject = subjects[0]

    # initialize recall trials DataFrame
    n_recall = np.sum([len(r) for r in recalls_list])
    recall = pd.DataFrame({'subject': subject,
                           'list': np.zeros(n_recall, dtype=int),
                           'trial_type': 'recall',
                           'position': np.zeros(n_recall, dtype=int),
                           'item': '',
                           'rt': np.zeros(n_recall, dtype=float)})

    # set basic information (list, item, position)
    n = 0
    for i, seq in enumerate(recalls_list):
        pool = study.loc[study['list'] == lists[i], 'item'].to_numpy()
        for j, pos in enumerate(seq):
            recall.loc[n, 'list'] = lists[i]
            recall.loc[n, 'item'] = pool[pos]
            recall.loc[n, 'position'] = j + 1
            recall.loc[n, 'rt'] = recalls_times[i][j]
            n += 1
    data = pd.concat((study, recall), axis=0, ignore_index=True)
    data = data.sort_values(['list', 'trial_type'], ascending=[True, False])
    return data


class CMRLBA(CMR):
    """CMR Distributed with Linear Ballistic Accumulators."""
    def prepare_sim(self, data, param):
        # 'dynamic' field on param
        # tells you where the dynamic param values are stored on the data struct
        study_key_names = ['input', 'item_index']
        recall_key_names = ['input', 'rt']
        if 'dynamic' in param:
            if 'study' in param['dynamic']:
                for pname in param['dynamic']['study'].keys():
                    datacol = param['dynamic']['study'][pname]
                    study_key_names = study_key_names + datacol
            if 'recall' in param['dynamic']:
                for pname in param['dynamic']['recall'].keys():
                    datacol = param['dynamic']['recall'][pname]
                    recall_key_names = recall_key_names + datacol
        study, recall = fit.prepare_lists(
            data, study_keys=study_key_names,
            recall_keys=recall_key_names, clean=True)
        return study, recall

    def generate_subject(self, study_data, param, patterns=None, weights=None,
                         **kwargs):
        study = fit.prepare_study(study_data,
                                  study_keys=['position', 'item_index'])
        n_item = len(study['position'][0])
        n_list = len(study['position'])
        list_param = cmr.prepare_list_param(n_item, param)

        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        recalls = []
        rts = []
        recall_time_limit = param['recall_time_limit']
        for i in range(n_list):
            net = cmr.init_dist_cmr(study['item_index'][i], scaled, param)
            net.study('item', study['position'][i], param['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            net.integrate('start', 0, param['B_start'])
            recall, rt = net.generate_recall_lba('item', recall_time_limit,
                                                 param['B_rec'], param['A'],
                                                 param['b'], param['s'], param['tau'])
            recalls.append(recall)
            rts.append(rt)
        data = add_recalls(study_data, recalls, rts)
        return data

    def likelihood_subject(self, study, recall, param, patterns=None,
                           weights=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        list_param = cmr.prepare_list_param(n_item, param)

        weights_param = network.unpack_weights(weights, param)
        scaled = network.prepare_patterns(patterns, weights_param)
        logl = 0
        n = 0
        for i in range(n_list):
            net = cmr.init_dist_cmr(study['item_index'][i], scaled, param)
            sparam = param.copy()
            if 'dynamic' in param:
                if 'study' in param['dynamic']:
                    sparam = cmr.update_dynamic_parameters(study, param, 'study', i)
            net.study('item', study['input'][i], sparam['B_enc'],
                      list_param['Lfc'], list_param['Lcf'])
            net.integrate('start', 0, param['B_start'])
            rparam = param.copy()
            if 'dynamic' in param:
                if 'recall' in param['dynamic']:
                    rparam = cmr.update_dynamic_parameters(recall, param, 'recall', i)
            p = self.p_recall_lba(net, 'item', recall['input'][i], recall['rt'][i], param['recall_time_limit'],
                             rparam['B_rec'], param['T'], param['A'], param['b'], param['s'], param['tau'])
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
            n += p.size
        return logl, n

    def p_recall_lba(self, net, segment, recalls, rts, recall_time_limit, B, T, A, b, s, tau):
        """Recall data struct must contain rt information for recall events."""
        amin = 0.000001
        # weights to use for recall (assume fixed during recall)
        rec_ind = net.f_ind[segment]
        n_item = net.n_f_segment[segment]
        w_cf = net.w_cf_exp[rec_ind, :] + net.w_cf_pre[rec_ind, :]

        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        # exclude = np.zeros(w_cf.shape[0], dtype=bool)
        p = np.zeros(len(recalls) + 1)
        for output, recall in enumerate(recalls):
            # project the current state of context; assume nonzero support
            # support = np.dot(w_cf, net.c)
            # support[support < amin] = amin

            # calculate item support (modifies f_in)
            operations.cue_item(
                rec_ind.start, n_item, net.w_cf_pre, net.w_cf_exp,
                net.w_ff_pre, net.w_ff_exp, net.f_in, net.c, exclude,
                np.asarray(recalls, dtype=np.dtype('i')), output
            )
            support = net.f_in[rec_ind]

            # TODO: allow some kind of non-linear scaling of support
            # scale based on choice parameter, set recalled items to zero
            #strength = np.exp((2 * support) / T)
            #strength[exclude] = 0
            strength = support
            strength[exclude] = 0

            # probability of this recall at this rt, conditional on not stopping
            if output > 0:
                resp_duration = rts[output] - rts[output-1]
            else:
                resp_duration = rts[output]
            p[output] = lba.response_pdf(resp_duration, recall, A, b, strength, s)

            # remove recalled item from competition
            exclude[recall] = 1

            # update context
            net.present(segment, recall, B)

        # probability of the stopping event
        # is the probability that no racers cross threshold before time is up
        time_left = recall_time_limit - rts[-1]

        p[len(recalls)] = lba.response_pdf(time_left, len(strength), A, b, strength, s)
        return p
