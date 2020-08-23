"""Test speed of likelihood calculation."""

import numpy as np

from cymr import cmr
from cymr import parameters


class TimeLikelihood(object):
    def setup(self):
        self.n_item = 24
        self.n_trial = 300
        self.n_item_total = 768
        self.n_dim = 300
        mat = np.random.normal(size=(self.n_item_total, self.n_dim))
        self.patterns = {'vector': {'loc': mat}}
        n_recall = np.random.choice(self.n_item, self.n_trial)
        items = np.arange(self.n_item_total)
        item_index = [np.random.choice(items, self.n_item, replace=False)
                      for i in range(self.n_trial)]
        inputs = [np.arange(self.n_item) for i in range(self.n_trial)]
        outputs = [np.random.choice(i, n, replace=False)
                   for i, n in zip(inputs, n_recall)]

        self.study = {'input': inputs, 'item_index': item_index}
        self.recall = {'input': outputs}

        param_def = parameters.Parameters()
        param_def.set_dependent(Dfc='1 - Lfc', Dcf='1 - Lcf')
        param_def.set_sublayers(f=['task'], c=['task'])
        param_def.set_weights('fc', {
            (('task', 'item'), ('task', 'item')): 'Dfc * loc',
        })
        param_def.set_weights('cf', {
            (('task', 'item'), ('task', 'item')): 'Dcf * loc',
        })
        self.param_def = param_def

        param = {
            'B_enc': .5, 'B_start': 0, 'B_rec': .8,
            'Lfc': .8, 'Lcf': .8, 'P1': 1, 'P2': 1,
            'T': 10, 'X1': .05, 'X2': .2
        }
        self.param = param_def.eval_dependent(param)
        self.model = cmr.CMRDistributed()

    def time_likelihood(self):
        self.model.likelihood_subject(
            self.study, self.recall, self.param, self.param_def, self.patterns
        )
