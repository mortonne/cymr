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

    Returns
    -------
    p_stop : numpy.array
        Probability of stopping for each output position.
    """
    p_stop = X1 * np.exp(X2 * np.arange(n_item + 1))
    p_stop[p_stop < pmin] = pmin
    p_stop[p_stop > (1 - pmin)] = 1 - pmin

    # after recalling all items, P(stop)=1 by definition
    p_stop[-1] = 1
    return p_stop


def config_loc_cmr(n_item):
    """
    Configure a localist CMR network.

    Parameters
    ----------
    n_item : int
        Number of item patterns to include in the network

    Returns
    -------
    param_def : cymr.parameters.Parameters
        Parameters object with configuration for the network.

    patterns : dict
        Patterns to place in the network.
    """
    items = np.arange(n_item)
    patterns = {'items': items, 'vector': {'loc': np.eye(n_item)}}
    param_def = parameters.Parameters()
    param_def.set_sublayers(f=['task'], c=['task'])
    param_def.set_weights('fc', {
        (('task', 'item'), ('task', 'item')): 'Afc + Dfc * loc'
    })
    param_def.set_weights('cf', {
        (('task', 'item'), ('task', 'item')): 'Acf + Dcf * loc'
    })
    return param_def, patterns


def init_network(param_def, patterns, param, item_index, remove_blank=False):
    """
    Initialize a network with pattern weights.

    Parameters
    ----------
    param_def : cymr.parameters.Parameters
        Parameters definition defining network sublayers and weights.

    patterns : dict
        Patterns to place in the network.

    param : dict of (str: float)
        Parameter values; used to initialize network weights.

    item_index : numpy.array
        Indices of item patterns to include in the network.

    remove_blank : bool
        If true, context units with zero weights will be removed.

    Returns
    -------
    net : cymr.network.Network
        Network initialized with pre-experimental weights.
    """
    # set item weights
    weights = param_def.eval_weights(patterns, param, item_index)

    if remove_blank:
        # remove context units that are zero for all items
        for connect in ['fc', 'cf']:
            for region, mat in weights[connect].items():
                include = np.any(mat != 0, 0)
                weights[connect][region] = mat[:, include]

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
    """
    Simulate study of a list.

    Parameters
    ----------
    param_def : cymr.parameters.Parameters
        Parameters definition defining network sublayers and weights.

    param : dict of (str: float)
        Parameter values; used to set context evolution and learning.

    item_index : numpy.array
        Indices of presented item patterns.

    item_input : numpy.array
        Input position of presented items.

    patterns : dict
        Item pattern vectors.

    Returns
    -------
    net : cymr.network.Network
        Network after simulation of the study phase.
    """
    net = init_network(param_def, patterns, param, item_index)
    net.update(('task', 'start', 0), net.c_sublayers)
    net.study(
        ('task', 'item'), item_input, net.c_sublayers, param['B_enc'],
        param['Lfc'], param['Lcf']
    )
    net.integrate(('task', 'start', 0), net.c_sublayers, param['B_start'])
    return net


def prepare_list_param(n_item, n_sub, param, param_def):
    """
    Prepare parameters that vary within list.

    Parameters
    ----------
    n_item : int
        Number of items in the list.

    n_sub : int
        Number of sublayers of context.

    param : dict of (str: float)
        Parameter values.

    param_def : cymr.parameters.Parameters
        Parameter definitions indicating sublayer parameters.

    Returns
    -------
    list_param : dict
        Parameters prepared for changes between sublayer and trials.
    """
    if 'c' in param_def.sublayer_param:
        # evaluate sublayer parameters
        param = param_def.eval_sublayer_param('c', param, n_item)

        # get the set of all sublayer parameters
        sub_param = set()
        for sublayer in param_def.sublayers['c']:
            for par in param_def.sublayer_param['c'][sublayer].keys():
                sub_param.add(par)
    else:
        sub_param = set()

    if 'Lfc' in sub_param:
        Lfc = param['Lfc']
    else:
        Lfc = np.tile(param['Lfc'], (n_item, n_sub)).astype(float)

    # apply primacy gradient to Lcf
    if 'Lcf' in sub_param:
        n_sub = param['Lcf'].shape[1]
        Lcf = np.zeros(param['Lcf'].shape)
        for i in range(n_sub):
            Lcf[:, i] = primacy(
                n_item, param['Lcf'][0, i], param['P1'], param['P2']
            )
    else:
        Lcf_trial = primacy(n_item, param['Lcf'], param['P1'], param['P2'])
        Lcf = np.tile(Lcf_trial[:, None], (1, n_sub))

    p_stop = p_stop_op(n_item, param['X1'], param['X2'])
    list_param = param.copy()
    list_param.update({'Lfc': Lfc, 'Lcf': Lcf, 'p_stop': p_stop})
    return list_param


class CMR(Recall):
    """
    Context Maintenance and Retrieval model.

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
                ('task', 'item'), net.c_sublayers, list_param['B_rec'],
                list_param['T'], list_param['p_stop']
            )
            recall_index = study['item_index'][i][recall_vec]
            recall_items = patterns['items'][recall_index]
            recalls_list.append(recall_items)
        return recalls_list

    def record_subject(self, study, recall, param, param_def=None,
                       patterns=None, remove_blank=False):
        n_item = len(study['input'][0])
        n_list = len(study['input'])
        if param_def is None:
            raise ValueError('Must provide a Parameters object.')
        n_sub = len(param_def.sublayers['c'])
        param = prepare_list_param(n_item, n_sub, param, param_def)

        study_state = []
        recall_state = []
        for i in range(n_list):
            # access the dynamic parameters needed for this list
            list_param = param.copy()
            list_param = param_def.get_dynamic(list_param, i)

            # initialize the network
            net = init_network(
                param_def, patterns, param, study['item_index'][i],
                remove_blank=remove_blank
            )
            net.update(('task', 'start', 0), net.c_sublayers)

            # record study phase
            item_list = study['input'][i].astype(int)
            list_study_state = net.record_study(
                ('task', 'item'), item_list, net.c_sublayers, param['B_enc'],
                list_param['Lfc'], list_param['Lcf']
            )
            net.integrate(('task', 'start', 0), net.c_sublayers, param['B_start'])

            # record recall phase
            list_recall_state = net.record_recall(
                ('task', 'item'), recall['input'][i], net.c_sublayers,
                param['B_rec'], param['T']
            )
            study_state.append(list_study_state)
            recall_state.append(list_recall_state)
        return study_state, recall_state
