"""Models of free recall."""

import numpy as np
from cymr.fit import Recall
from cymr import fit
from cymr import network
from cymr import parameters


def primacy(n_item, L, P1, P2):
    """
    Primacy gradient in learning rate.

    Parameters
    ----------
    n_item : int
        Number of items in study list.

    L : float
        Base learning rate. Asymptote of gradient for later positions.

    P1 : float
        Additional learning for first item.

    P2 : float
        Decay rate for primacy gradient.

    Returns
    -------
    rate : numpy.array
        Learning rate for each serial position.
    """
    position = np.arange(n_item)
    rate = L + (P1 * np.exp(-P2 * position))
    return rate


def p_stop_op(n_item, X1, X2, pmin=0.000001):
    """
    Probability of stopping based on output position.

    Parameters
    ----------
    n_item : int
        Number of items available for recall.

    X1 : float
        Probability of not recalling any items.

    X2 : float
        Shape parameter of exponential function increasing stop
        probability by output position.

    pmin : float, optional
        Minimum probability of stopping recall at any position.
    """
    p_stop = X1 * np.exp(X2 * np.arange(n_item + 1))
    p_stop[p_stop < pmin] = pmin
    p_stop[p_stop > (1 - pmin)] = 1 - pmin

    # after recalling all items, P(stop)=1 by definition
    p_stop[-1] = 1
    return p_stop


def init_loc_cmr(n_item, param):
    """Initialize localist CMR for one list."""
    f_segment = {'task': {'item': n_item, 'start': 1}}
    c_segment = {'task': {'item': n_item, 'start': 1}}

    net = network.Network(f_segment, c_segment)
    net.add_pre_weights('fc', ('task', 'item'), ('task', 'item'),
                        np.eye(n_item), param['Dfc'], param['Afc'])
    net.add_pre_weights('cf', ('task', 'item'), ('task', 'item'),
                        np.eye(n_item), param['Dcf'], param['Acf'])
    net.add_pre_weights('fc', ('task', 'start'), ('task', 'start'), 1)
    net.update(('task', 'start', 0), 'task')
    return net


def init_network(param_def, patterns, param, item_index):
    """Initialize a network with pattern weights."""
    # set item weights
    weights = param_def.eval_weights(patterns, param, item_index)

    # set task units
    for f_sublayer in param_def.sublayers['f']:
        for c_sublayer in param_def.sublayers['c']:
            region = ((f_sublayer, 'start'), (c_sublayer, 'start'))
            weights['fc'][region] = np.array([[1]])
            weights['cf'][region] = np.array([[1]])

    # get all segment definitions
    f_segments = {}
    c_segments = {}
    for region, mat in weights['fc'].items():
        f_region, c_region = region
        f_sublayer, f_segment = f_region
        c_sublayer, c_segment = c_region
        if f_sublayer not in f_segments:
            f_segments[f_sublayer] = {}
        if c_sublayer not in c_segments:
            c_segments[c_sublayer] = {}
        f_segments[f_sublayer][f_segment] = mat.shape[0]
        c_segments[c_sublayer][c_segment] = mat.shape[1]

    # initialize the network
    net = network.Network(f_segments, c_segments)
    for connect in weights.keys():
        for region, mat in weights[connect].items():
            if connect == 'ff':
                f_segment = region
                c_segment = None
            else:
                f_segment, c_segment = region
            net.add_pre_weights(connect, f_segment, c_segment, mat)
    return net


def study_list(param_def, param, item_index, item_input, patterns):
    """Simulate study of a list."""
    net = init_network(param_def, patterns, param, item_index)
    net.update(('task', 'start', 0), net.c_sublayers)
    net.study(
        ('task', 'item'), item_input, net.c_sublayers, param['B_enc'],
        param['Lfc'], param['Lcf']
    )
    net.integrate(('task', 'start', 0), net.c_sublayers, param['B_start'])
    return net


def prepare_list_param(n_item, n_sub, param, param_def):
    """Prepare parameters that vary within list."""
    Lfc = np.tile(param['Lfc'], (n_item, n_sub)).astype(float)
    Lcf_trial = primacy(n_item, param['Lcf'], param['P1'], param['P2'])
    Lcf = np.tile(Lcf_trial[:, None], (1, n_sub))
    p_stop = p_stop_op(n_item, param['X1'], param['X2'])
    list_param = param.copy()
    list_param.update({'Lfc': Lfc, 'Lcf': Lcf, 'p_stop': p_stop})
    if 'c' in param_def.sublayer_param:
        list_param = param_def.eval_sublayer_param('c', list_param, n_item)
    return list_param


class CMR(Recall):

    def prepare_sim(self, data, study_keys=None, recall_keys=None):
        study, recall = fit.prepare_lists(data, study_keys=['input'],
                                          recall_keys=['input'], clean=True)
        return study, recall

    def likelihood_subject(self, study, recall, param, param_def=None,
                           patterns=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        n_sub = 1
        if param_def is None:
            param_def = parameters.Parameters()
        param_def.set_sublayers(f=['task'], c=['task'])
        trial_param = prepare_list_param(n_item, n_sub, param, param_def)
        net_init = init_loc_cmr(n_item, param)
        logl = 0
        n = 0
        for i in range(n_list):
            net = net_init.copy()
            list_param = param.copy()
            if param_def is not None:
                list_param = param_def.get_dynamic(list_param, i)
            net.study(
                ('task', 'item'), study['input'][i], 'task',
                list_param['B_enc'], trial_param['Lfc'], trial_param['Lcf']
            )
            p = net.p_recall(
                ('task', 'item'), recall['input'][i], 'task', list_param['B_rec'],
                list_param['T'], trial_param['p_stop']
            )
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
            n += p.size
        return logl, n

    def generate_subject(self, study, recall, param, param_def=None,
                         patterns=None, **kwargs):
        # study = fit.prepare_study(study_data, study_keys=['position'])
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        n_sub = 1
        if param_def is None:
            param_def = parameters.Parameters()
        param_def.set_sublayers(f=['task'], c=['task'])
        trial_param = prepare_list_param(n_item, n_sub, param, param_def)

        net_init = init_loc_cmr(n_item, param)
        recalls_list = []
        for i in range(n_list):
            net = net_init.copy()
            list_param = param.copy()
            if param_def is not None:
                list_param = param_def.get_dynamic(list_param, i)
            net.study(
                ('task', 'item'), study['input'][i], 'task', list_param['B_enc'],
                trial_param['Lfc'], trial_param['Lcf']
            )
            recall_vec = net.generate_recall(
                ('task', 'item'), 'task', list_param['B_rec'],
                list_param['T'], trial_param['p_stop']
            )
            recalls_list.append(recall_vec)
        return recalls_list

    def record_network(self, data, param):
        study, recall = self.prepare_sim(data)
        n_item = len(study['input'][0])
        n_sub = 1
        param_def = parameters.Parameters()
        param_def.set_sublayers(f=['task'], c=['task'])
        list_param = prepare_list_param(n_item, n_sub, param, param_def)
        net_init = init_loc_cmr(n_item, param)
        n_list = len(study['input'])

        net_state = []
        for i in range(n_list):
            net = net_init.copy()
            item_list = study['input'][i].astype(int)
            state = net.record_study(
                ('task', 'item'), item_list, 'task', param['B_enc'],
                list_param['Lfc'], list_param['Lcf']
            )
            rec = net.record_recall(
                ('task', 'item'), recall['input'][i], 'task',
                param['B_rec'], param['T']
            )
            state.extend(rec)
            net_state.append(state)
        return net_state


class CMRDistributed(Recall):
    """
    Context Maintenance and Retrieval-Distributed model.

    **Model Parameters**

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

    **Parameter definition objects**

    Parameters objects are used to indicate sublayers to include
    in the network and to indicate how network weights should be
    initialized. The :code:`sublayers` and :code:`weights` attributes
    must be set.

    Parameters objects may also be used to define parameters that depend
    on other parameters and/or dynamic parameters that depend on columns
    of the input data.

    Finally, Parameters objects are used to define searches, using the
    :code:`fixed` and :code:`free` attributes.

    **Model Patterns**

    Patterns are used to define connections between the item and
    context layers and direct connections between items. Connections
    may be orthonormal as in many published variants of CMR, or they
    may be distributed, overlapping patterns.

    Patterns may include :code:`'vector'` and/or :code:`'similarity'` matrices.
    Vector representations are used to set the :math:`M^{FC}_{pre}`
    and :math:`M^{CF}_{pre}` matrices, while similarity matrices are
    used to set the :math:`M^{FF}_{pre}` matrix.

    Vector and similarity values are dicts of (feature: array) specifying
    an array for one or more named features, with an
    [items x units] array for vector representations, or
    [items x items] for similarity matrices.
    """

    def prepare_sim(self, data, study_keys=None, recall_keys=None):
        study_base = ['input', 'item_index']
        if study_keys is None:
            study_keys = study_base
        else:
            # only add the base key if it isn't already in there
            for term in study_base:
                if term not in study_keys:
                    study_keys += [term]
        recall_base = ['input']
        if recall_keys is None:
            recall_keys = recall_base
        else:
            for term in recall_base:
                if term not in recall_keys:
                    recall_keys += [term]

        study, recall = fit.prepare_lists(
            data, study_keys=study_keys, recall_keys=recall_keys, clean=True
        )
        return study, recall

    def likelihood_subject(self, study, recall, param, param_def=None,
                           patterns=None):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        if param_def is None:
            raise ValueError('Must provide a Parameters object.')
        n_sub = len(param_def.sublayers['c'])
        param = prepare_list_param(n_item, n_sub, param, param_def)

        logl = 0
        n = 0
        for i in range(n_list):
            # access the dynamic parameters needed for this list
            list_param = param.copy()
            list_param = param_def.get_dynamic(list_param, i)

            # simulate study
            net = study_list(
                param_def, list_param, study['item_index'][i],
                study['input'][i], patterns
            )

            # get recall probabilities
            p = net.p_recall(
                ('task', 'item'), recall['input'][i], net.c_sublayers,
                list_param['B_rec'], list_param['T'], list_param['p_stop']
            )
            if np.any(np.isnan(p)) or np.any((p <= 0) | (p >= 1)):
                logl = -10e6
                break
            logl += np.sum(np.log(p))
            n += p.size
        return logl, n

    def generate_subject(self, study, recall, param, param_def=None,
                         patterns=None, **kwargs):

        n_item = len(study['input'][0])
        n_list = len(study['input'])
        if param_def is None:
            raise ValueError('Must provide a Parameters object.')
        n_sub = len(param_def.sublayers['c'])
        param = prepare_list_param(n_item, n_sub, param, param_def)

        recalls_list = []
        for i in range(n_list):
            # access the dynamic parameters needed for this list
            list_param = param.copy()
            list_param = param_def.get_dynamic(list_param, i)

            # simulate study
            net = study_list(
                param_def, list_param, study['item_index'][i],
                study['input'][i], patterns
            )

            # simulate recall
            recall_vec = net.generate_recall(
                ('task', 'item'), 'task', list_param['B_rec'],
                list_param['T'], list_param['p_stop']
            )
            recalls_list.append(recall_vec)
        return recalls_list

    def record_network(self, data, param, param_def=None, patterns=None,
                       remove_blank=False):
        study, recall = self.prepare_sim(data)
        n_item = len(study['input'][0])
        if param_def is None:
            raise ValueError('Must provide a Parameters object.')
        n_sub = len(param_def.sublayers['c'])
        list_param = prepare_list_param(n_item, n_sub, param, param_def)
        n_list = len(study['input'])

        net_state = []
        for i in range(n_list):
            if remove_blank:
                # include = np.any(scaled['fcf'][study['item_index'][i]] != 0, 0)
                # scaled['fcf'] = scaled['fcf'][:, include]
                raise ValueError('remove_blank option currently unsupported.')
            net = init_network(param_def, patterns, param, study['item_index'][i])
            item_list = study['input'][i].astype(int)
            state = net.record_study(
                ('task', 'item'), item_list, 'task', param['B_enc'],
                list_param['Lfc'], list_param['Lcf']
            )
            net.integrate(('task', 'start', 0), 'task', param['B_start'])
            rec = net.record_recall(
                ('task', 'item'), recall['input'][i], 'task',
                param['B_rec'], param['T']
            )
            state.extend(rec)
            net_state.append(state)
        return net_state
