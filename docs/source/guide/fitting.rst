
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

    from cymr import fit
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

dependent
    Parameters that are derived from other parameters. These parameters
    are specified using an expression that generates them from other
    parameters.

We can organize these things by creating a Parameters object. To speed
things up, we'll fix almost all parameters and just fit one,
:math:`\beta_\mathrm{enc}`.

.. ipython:: python

    par = fit.Parameters()
    par.add_fixed(Afc=0, Acf=0, Aff=0, Dff=1, T=0.1,
                  Lfc=0.15, Lcf=0.15, P1=0.2, P2=2,
                  B_start=0.3, B_rec=0.9, X1=0.001, X2=0.25)
    par.add_free(B_enc=(0, 1))
    par.add_dependent(Dfc='1 - Lfc', Dcf='1 - Lcf')

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

Patterns may include multiple components that may be weighted differently.
Weight parameters are used to set the weighting of each component. Here,
we only have one component, which will have a fixed weight of 1.

.. ipython:: python

    par.add_weights('fcf', {'loc': 'w_loc'})
    par.add_fixed(w_loc=1)

Parameter Search
~~~~~~~~~~~~~~~~

Finally, we can run the search. For speed, we'll set the tolerance to
be pretty high (0.1); normally this should be much lower to ensure
that the search converges.

.. ipython:: python

    from cymr import cmr
    model = cmr.CMRDistributed()
    results = model.fit_indiv(data, par.fixed, par.free,
                              dependent=par.dependent, tol=0.1,
                              patterns=patterns, weights=par.weights)
    results

The results give the complete set of parameters, including fixed
parameters, optimized free parameters, and dependent parameters. It
also includes fields with statistics relevant to the search:

logl
    Total log likelihood for each participant.

n
    Number of data points fit.

k
    Number of free parameters.
