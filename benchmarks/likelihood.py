"""Test speed of likelihood calculation."""

import numpy as np

from cymr import fit
from cymr import cmr
from cymr import parameters


class TimeLikelihood(object):
    def setup(self):
        self.data = fit.sample_data('Morton2013_mixed')

        n_items = 768
        self.patterns = {'vector': {'loc': np.eye(n_items)}}

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
        self.model = cmr.CMR()

    def time_likelihood(self):
        self.model.likelihood(
            self.data, self.param, None, self.param_def, self.patterns
        )
