
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

===============
Fitting a model
===============

We can fit a model to individual participant data in a free-recall dataset
by maximizing the probability of the data according to the model. This involves
using a search algorithm to adjust the model parameters until the probability,
or likelihood (see :doc:`/guide/evaluation`), of the data is maximized.

First, load some sample data to fit:

.. ipython:: python

    from cymr import fit, cmr
    data = fit.sample_data('Morton2013_mixed').query('subject <= 3')

Search Definition
~~~~~~~~~~~~~~~~~

Next, we need to define our search parameters. There are two types
of parameters used specifically for searches:

fixed
    Parameters that have a fixed value. These parameters are not searched.

free
    Parameters that may vary to fit a dataset. For a search, must specify
    a range to be searched over.

We'll also use two other types of parameters that set properties of the model
based on a given parameter set:

dependent
    Parameters that are derived from other parameters. These parameters
    are specified using an expression that generates them from other
    parameters.

weights
    Parameters that define weighting of different patterns in the model.

We can organize these things by creating a Parameters object. To run
a simple and fast search, we'll fix almost all parameters and just fit one,
:math:`\beta_\mathrm{enc}`. For a real project, you may want to free other
parameters also to fit individual differences in the primacy effect, temporal
clustering, etc.

.. ipython:: python

    par = cmr.CMRParameters()
    par.set_fixed(T=0.1, Lfc=0.15, Lcf=0.15, P1=0.2, P2=2,
                  B_start=0.3, B_rec=0.9, X1=0.001, X2=0.25)
    par.set_free(B_enc=(0, 1))
    par.set_dependent(Dfc='1 - Lfc', Dcf='1 - Lcf')

To simulate free recall using the context maintenance and retrieval (CMR) model, we must first
define pre-experimental weights for the network. For this example, we'll define
localist patterns, which are distinct for each presented item. They can be
represented by an identity matrix with one entry for each item. See
:doc:`/guide/evaluation` for details.

.. ipython:: python

    n_items = 768
    study = data.query("trial_type == 'study'")
    items = study.groupby('item_index')['item'].first().to_numpy()
    patterns = {'items': items, 'vector': {'loc': np.eye(n_items)}}
    par.set_sublayers(f=['task'], c=['task'])
    weights = {(('task', 'item'), ('task', 'item')): 'loc'}
    par.set_weights('fc', weights)
    par.set_weights('cf', weights)

We can print the parameter definition to get an overview of the settings.

.. ipython:: python

    print(par)

The :py:meth:`~cymr.cmr.CMRParameters.to_json` method of
:py:class:`~cymr.cmr.CMRParameters` can be used to save out parameter
definitions to a file. The output file uses JSON format, which is
both human- and machine-readable and can be loaded later to restore
search settings:

.. ipython:: python

    par.to_json('parameters.json')
    restored = cmr.read_config('parameters.json')

Parameter Search
~~~~~~~~~~~~~~~~

Finally, we can run the search. Parameters will be optimized separately
for each participant. For speed, we'll set the tolerance to
be pretty high (0.1); normally this should be much lower to ensure
that the search converges.

.. ipython:: python

    from cymr import cmr
    model = cmr.CMR()
    results = model.fit_indiv(data, par, patterns=patterns, tol=0.1)
    results[['B_enc', 'logl', 'n', 'k']]

The results give the complete set of parameters, including fixed
parameters, optimized free parameters, and dependent parameters. It
also includes fields with statistics relevant to the search:

logl
    Total log likelihood for each participant. Greater (i.e., less negative)
    values indicate better fit.

n
    Number of data points fit.

k
    Number of free parameters.
