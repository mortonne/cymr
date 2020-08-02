
.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use('default')
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1

    pd.options.display.max_rows = 15

==================
Evaluating a model
==================

For a given model and set of parameters, we can measure the overall
fit to a dataset by calculating the log likelihood. The *likelihood*
is the probability of the data according to the model. Probability is
calculated for full recall sequences and then multiplied for each recall
sequence to obtain an overall probability of the data. In practice, this
leads to extremely small probabilities, which may be difficult for the
computer to calculate. Therefore, we use log probabilities to avoid this
problem. For both likelihoods and log likelihood, greater values indicate
a better fit of the model to the data.

First, load some sample data:

.. ipython:: python

    from cymr import fit, parameters
    data = fit.sample_data('Morton2013_mixed').query('subject <= 3')

Patterns and Weights
~~~~~~~~~~~~~~~~~~~~

To simulate free recall using the CMR-Distributed model, we must first
define pre-experimental weights for the network. For this example, we'll define
localist patterns, which are distinct for each presented item. They can be
represented by an identity matrix with one entry for each item.

.. ipython:: python

    n_items = 768
    loc_patterns = np.eye(n_items)

To indicate where the patterns should be used in the network, they are
specified as :code:`vector` (for the :math:`\mathrm{M}^{FC}` and/or
:math:`\mathrm{M}^{CF}` matrices) or :code:`similarity`
(for the :math:`\mathrm{M}^{FF}` matrix). We also label each pattern
with a name; here, we'll refer to the localist patterns as :code:`'loc'`.

.. ipython:: python

    patterns = {'vector': {'loc': loc_patterns}}

Parameters
~~~~~~~~~~

:py:class:`~cymr.parameters.Parameters` objects define how parameter values will be
interpreted. One use of them is to define how weights in the model network
are set.

Patterns may include multiple components that may be weighted differently.
Weight parameters are used to set the weighting of each component. Here,
we only have one component, which we assign a weight based on the value
of the :code:`w_loc` parameter.

.. ipython:: python

    param_def = parameters.Parameters()
    param_def.set_weights('fcf', {'loc': 'w_loc'})

Finally, we define the parameters that we want to evaluate, by creating
a dictionary with a name and value for each parameter. We'll get a
different log likelihood for each parameter set. For a model to be
evaluated, all parameters expected by that model must be defined,
including any parameters used for setting weights (here, :code:`w_loc`).

.. ipython:: python

    param = {
        'B_enc': 0.7,
        'B_start': 0.3,
        'B_rec': 0.9,
        'w_loc': 1,
        'Afc': 0,
        'Dfc': 0.85,
        'Acf': 1,
        'Dcf': 0.85,
        'Aff': 0,
        'Dff': 1,
        'Lfc': 0.15,
        'Lcf': 0.15,
        'P1': 0.2,
        'P2': 2,
        'T': 0.1,
        'X1': 0.001,
        'X2': 0.25
    }

Evaluating log likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~

Define a model (here, cmr.CMRDistributed) and use :py:meth:`~cymr.fit.Recall.likelihood`
to evaluate the log likelihood of the observed data according to that model
and these parameter values. Greater (i.e., less negative) log likelihood values
indicate a better fit. In :doc:`/guide/fitting`, we'll use a parameter search to estimate
the best-fitting parameters for a model.

.. ipython:: python

    from cymr import cmr
    model = cmr.CMRDistributed()
    logl, n = model.likelihood(data, param, param_def=param_def, patterns=patterns)
    print(f'{n} data points evaluated.')
    print(f'Log likelihood is: {logl:.4f}')
