"""Represent interactions between context and item layers."""

import numpy as np
from cymr import operations


class Network(object):

    def __init__(self, n_c, n_f):
        self.n_f = n_f
        self.n_c = n_c
        self.f = np.zeros(n_f)
        self.c = np.zeros(n_c)
        self.c_in = np.zeros(n_c)
        self.w_fc_pre = np.zeros((n_c, n_f))
        self.w_fc_exp = np.zeros((n_c, n_f))
        self.w_cf_pre = np.zeros((n_f, n_c))
        self.w_cf_exp = np.zeros((n_f, n_c))

    def __repr__(self):
        s_f = 'f:\n' + self.f.__str__()
        s_c = 'c:\n' + self.c.__str__()
        s_fc_pre = 'W pre [fc]:\n' + self.w_fc_pre.__str__()
        s_fc_exp = 'W exp [fc]:\n' + self.w_fc_exp.__str__()
        s_cf_pre = 'W pre [cf]:\n' + self.w_cf_pre.__str__()
        s_cf_exp = 'W exp [cf]:\n' + self.w_cf_exp.__str__()
        s = '\n\n'.join([s_f, s_c, s_fc_pre, s_fc_exp, s_cf_pre, s_cf_exp])
        return s

    def add_pre_weights(self, weights, slope=1, intercept=0):
        n_c, n_f = weights.shape
        assert n_c == self.n_c
        scaled = intercept + slope * weights
        self.w_cf_pre[:n_f, :] = scaled.T
        self.w_fc_pre[:, :n_f] = scaled

    def present_item(self, item, B):
        self.c_in = self.w_fc_pre[:, item].copy()
        self.c_in /= np.linalg.norm(self.c_in, ord=2)
        operations.update(self.c, self.c_in, B)
