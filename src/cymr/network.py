"""Represent interactions between context and item layers."""

import numpy as np
from cymr import operations


def p_stop_op(n_item, X1, X2, pmin=0.000001):
    """Probability of stopping based on output position."""
    p_stop = X1 * np.exp(X2 * np.arange(n_item + 1))
    p_stop[p_stop < pmin] = pmin

    # after recalling all items, P(stop)=1 by definition
    p_stop[-1] = 1
    return p_stop


class Network(object):

    def __init__(self, segments):
        n_f = 0
        n_c = 0
        self.f_ind = {}
        self.c_ind = {}
        for name, (s_f, s_c) in segments.items():
            self.f_ind[name] = slice(n_f, n_f + s_f)
            self.c_ind[name] = slice(n_c, n_c + s_c)
            n_f += s_f
            n_c += s_c
        self.f_ind['all'] = slice(0, n_f)
        self.c_ind['all'] = slice(0, n_c)

        self.n_f = n_f
        self.n_c = n_c
        self.f = np.zeros(n_f)
        self.c = np.zeros(n_c)
        self.c_in = np.zeros(n_c)
        self.w_fc_pre = np.zeros((n_f, n_c))
        self.w_fc_exp = np.zeros((n_f, n_c))
        self.w_cf_pre = np.zeros((n_f, n_c))
        self.w_cf_exp = np.zeros((n_f, n_c))

    def __repr__(self):
        np.set_printoptions(precision=4, floatmode='fixed', sign=' ')
        s_f = 'f:\n' + self.f.__str__()
        s_c = 'c:\n' + self.c.__str__()
        s_fc_pre = 'W pre [fc]:\n' + self.w_fc_pre.__str__()
        s_fc_exp = 'W exp [fc]:\n' + self.w_fc_exp.__str__()
        s_cf_pre = 'W pre [cf]:\n' + self.w_cf_pre.__str__()
        s_cf_exp = 'W exp [cf]:\n' + self.w_cf_exp.__str__()
        s = '\n\n'.join([s_f, s_c, s_fc_pre, s_fc_exp, s_cf_pre, s_cf_exp])
        return s

    def get_slices(self, region):
        f_ind = self.f_ind[region[0]]
        c_ind = self.c_ind[region[1]]
        return f_ind, c_ind

    def get_ind(self, layer, region, item):
        if layer == 'f':
            ind = self.f_ind[region].start + item
        elif layer == 'c':
            ind = self.c_ind[region].start + item
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def add_pre_weights(self, weights, region, slope=1, intercept=0):
        scaled = intercept + slope * weights
        f_ind, c_ind = self.get_slices(region)
        self.w_cf_pre[f_ind, c_ind] = scaled
        self.w_fc_pre[f_ind, c_ind] = scaled

    def update(self, segment, item):
        ind = self.f_ind[segment].start + item
        self.c_in = self.w_fc_pre[ind, :].copy() + self.w_fc_exp[ind, :].copy()
        self.c_in /= np.linalg.norm(self.c_in, ord=2)
        self.c = self.c_in.copy()

    def present(self, segment, item, B):
        ind = self.f_ind[segment].start + item
        operations.present(self.w_fc_exp, self.w_fc_pre,
                           self.c, self.c_in, self.f, ind, B)

    def learn(self, connect, segment, item, L):
        ind = self.c_ind[segment]
        if connect == 'fc':
            self.w_fc_exp[item, ind] += self.c[ind] * L
        elif connect == 'cf':
            self.w_cf_exp[item, ind] += self.c[ind] * L
        else:
            raise ValueError(f'Invalid connection: {connect}')

    def p_recall_cython(self, segment, recalls, B, T, p_stop, amin=0.000001):
        rec_ind = self.f_ind[segment]
        n_item = rec_ind.stop - rec_ind.start
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        p = np.zeros(len(recalls) + 1)
        recalls = np.array(recalls, dtype=np.dtype('i'))
        support = np.zeros(n_item)
        operations.p_recall(rec_ind.start, n_item, recalls,
                            self.w_fc_exp, self.w_fc_pre,
                            self.w_cf_exp, self.w_cf_pre, self.c, self.c_in,
                            exclude, amin, B, T, p_stop, support, p)
        return p

    def p_recall(self, segment, recalls, B, T, p_stop, amin=0.000001):
        # weights to use for recall (assume fixed during recall)
        rec_ind = self.f_ind[segment]
        w_cf = self.w_cf_exp[rec_ind, :] + self.w_cf_pre[rec_ind, :]

        exclude = np.zeros(w_cf.shape[0], dtype=bool)
        p = np.zeros(len(recalls) + 1)
        for output, recall in enumerate(recalls):
            # project the current state of context; assume nonzero support
            support = np.dot(w_cf, self.c)
            support[support < amin] = amin

            # scale based on choice parameter, set recalled items to zero
            strength = np.exp((2 * support) / T)
            strength[exclude] = 0

            # probability of this recall, conditional on not stopping
            p[output] = ((strength[recall] / np.sum(strength)) *
                         (1 - p_stop[output]))

            # remove recalled item from competition
            exclude[recall] = True

            # update context
            self.present(segment, recall, B)

        # probability of the stopping event
        p[len(recalls)] = p_stop[len(recalls)]
        return p
