
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

First, load some sample data to fit:

.. ipython:: python

    from cymr import fit, parameters
    data = fit.sample_data('Morton2013_mixed').query('subject <= 3')

Search Definition
~~~~~~~~~~~~~~~~~

Next, we need to define our search parameters. There are three types
of parameters used for searches:

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

    par = parameters.Parameters()
    par.set_fixed(Afc=0, Acf=0, Aff=0, Dff=1, T=0.1,
                  Lfc=0.15, Lcf=0.15, P1=0.2, P2=2,
                  B_start=0.3, B_rec=0.9, X1=0.001, X2=0.25)
    par.set_free(B_enc=(0, 1))
    par.set_dependent(Dfc='1 - Lfc', Dcf='1 - Lcf')

To simulate free recall using the CMR-Distributed model, we must first
define pre-experimental weights for the network. For this example, we'll define
localist patterns, which are distinct for each presented item. They can be
represented by an identity matrix with one entry for each item. See
:doc:`/guide/evaluation` for details.

.. ipython:: python

    n_items = 768
    loc_patterns = np.eye(n_items)
    patterns = {'vector': {'loc': loc_patterns}}
    par.set_weights('fcf', {'loc': 'w_loc'})
    par.set_fixed(w_loc=1)

Parameter Search
~~~~~~~~~~~~~~~~

Finally, we can run the search. For speed, we'll set the tolerance to
be pretty high (0.1); normally this should be much lower to ensure
that the search converges.

.. ipython:: python

    from cymr import cmr
    model = cmr.CMRDistributed()
    results = model.fit_indiv(data, par, patterns=patterns, tol=0.1)
    results[['B_enc', 'logl', 'n', 'k']]

The results give the complete set of parameters, including fixed
parameters, optimized free parameters, and dependent parameters. It
also includes fields with statistics relevant to the search:

logl
    Total log likelihood for each participant.

n
    Number of data points fit.

k
    Number of free parameters.
