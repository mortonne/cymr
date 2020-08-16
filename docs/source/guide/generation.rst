
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

=========================
Generating simulated data
=========================

While fitting a model can be accomplished by evaluating the log
likelihood of the data (see :doc:`/guide/fitting`), to interpret
the behavior of a model with a given set of parameters it's important
to generate simulated data. These data can then be analyzed the same
way as real data.

Loading data to simulate
~~~~~~~~~~~~~~~~~~~~~~~~

First, we need an experiment to run. This is specified using
psifr DataFrame format. We'll use data from a sample experiment.
Only the study trials are needed in this case. They'll specify
the order in which items are presented in each list during the
simulation.

.. ipython:: python

    from cymr import fit, parameters
    from psifr import fr
    data = fit.sample_data('Morton2013_mixed').query('subject == 1')
    fr.filter_data(data, trial_type='study')

.. note:: It's also possible to use columns of the DataFrame
    to set dynamic parameter values that vary over trial. For
    example, if some lists have a distraction task, you could
    have context integration rate vary with the amount of distraction.
    See dynamic parameter methods in :doc:`/api/parameters`.

Setting parameters
~~~~~~~~~~~~~~~~~~

Next, we define parameters for the simulation. Often these will be
taken from a parameter fit (see :doc:`/guide/fitting`). Here, we'll
just define the parameters we want directly. We also need to create
a :py:class:`~cymr.parameters.Parameters` object to define how the model
patterns are used.

.. ipython:: python

    param = {
        'B_enc': 0.6,
        'B_start': 0.3,
        'B_rec': 0.8,
        'Afc': 0,
        'Dfc': 0.85,
        'Acf': 1,
        'Dcf': 0.85,
        'Aff': 0,
        'Dff': 1,
        'Lfc': 0.15,
        'Lcf': 0.15,
        'P1': 0.8,
        'P2': 1,
        'T': 0.1,
        'X1': 0.001,
        'X2': 0.35
    }
    patterns = {'vector': {'loc': np.eye(768)}}
    param_def = parameters.Parameters()
    weights = {(('task', 'item'), ('task', 'item')): 'loc'}
    param_def.set_weights('fc', weights)
    param_def.set_weights('cf', weights)

Running a simulation
~~~~~~~~~~~~~~~~~~~~

We can then use the data, which define the items to study and recall
on each list, with the parameters and patterns, to general simulated
data using the CMR model. We'll repeat the simulation five times to
get a stable estimate of the model's behavior in this task.

.. ipython:: python

    from cymr import cmr
    model = cmr.CMRDistributed()
    sim = model.generate(data, param, param_def=param_def, patterns=patterns, n_rep=5)

Analying simulated data
~~~~~~~~~~~~~~~~~~~~~~~

We can then use the Psifr package to score and analyze the simulated
data just as we would real data. First, we score the data to prepare
it for analysis. This generates a new DataFrame that merges study and recall
events for each list:

.. ipython:: python

    sim_data = fr.merge_free_recall(sim)
    sim_data

Next, we can plot recall as a function of serial position:

.. ipython:: python

    recall = fr.spc(sim_data)

    @savefig spc.png
    g = fr.plot_spc(recall)

We can also analyze the order in which items are recalled by calculating
conditional response probability as a function of lag:

.. ipython:: python

    crp = fr.lag_crp(sim_data)

    @savefig lag_crp.png
    g = fr.plot_lag_crp(crp)

Peaks at short lags (e.g., -1, +1) indicate a tendency for items in nearby
serial positions to be recalled successively.

See :py:mod:`psifr.fr` for more analyses that you can run using Psifr.
