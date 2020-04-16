"""Test speed of likelihood calculation."""

import numpy as np
from cymr import network


class TimeLikelihood(object):
    def setup(self):
        self.B = .8
        self.L = 1
        self.T = 10
        self.X1 = .05
        self.X2 = 1
        self.n_item = 24
        self.n_trial = 300
        self.n_item_total = 768
        self.n_dim = 300
        self.patterns = np.random.normal(size=(self.n_item_total, self.n_dim))
        n_recall = np.random.choice(self.n_item, self.n_trial)
        items = np.arange(self.n_item_total)
        self.study = [np.random.choice(items, self.n_item, replace=False)
                      for i in range(self.n_trial)]
        inputs = np.arange(self.n_item)
        self.recall = [np.random.choice(inputs, n, replace=False)
                       for n in n_recall]

    def time_likelihood(self):
        for study, recall in zip(self.study, self.recall):
            net = network.Network({'item': (self.n_item, self.n_dim),
                                   'start': (1, 1)})
            item_patterns = self.patterns[study, :]
            net.add_pre_weights(item_patterns, ('item', 'item'))
            net.add_pre_weights(1, ('start', 'start'))
            net.update('start', 0)
            item_list = np.arange(len(study), dtype=int)
            net.study('item', item_list, self.B, self.L, self.L)
            p_stop = network.p_stop_op(self.n_item, self.X1, self.X2)
            p = net.p_recall('item', recall, self.B, self.T, p_stop)
