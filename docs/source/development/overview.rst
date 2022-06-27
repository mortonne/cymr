===========
Development
===========

Architecture
~~~~~~~~~~~~

The code is divided into three levels:

network
    The low-level network code represents layers and connection
    weights as Numpy arrays. It also defines operations (generally
    written in Cython for speed) on the network, such as studying
    an item or simulating free recall.

model
    Multiple models can be defined using the same network code. A
    model is defined in terms of likelihood calculation and/or
    simulation. Each model defines a set of parameters that affect
    the behavior of that model.

framework
    Defines a set of models to consider in the context of a project.
    Multiple models may be considered and compared in their ability
    to fit a dataset.

The cymr package defines the network and model levels. It also includes
infrastructure to help with implementing model frameworks. The cmr_cfr
package is an example of implementing a framework.

Defining a model
~~~~~~~~~~~~~~~~

Models should inherit from :py:class:`cymr.fit.Recall`, which defines
high-level methods for fitting data and generating simulated data. The
:py:class:`~cymr.fit.Recall` class also includes methods for running a
simple parameter recovery test and for running diagnostic parameter sweeps
to check how model simulations change when parameters are varied.

Preparing data for simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For any model, you must define a method called :code:`prepare_sim`, which
converts a Psifr DataFrame into list-format data. List-format data splits
data into study and recall phases. Each phase has a dictionary with entries
for different information about each trial; for example, the input position
of each item or the index of each item in the stimulus pool. Each entry in the
dictionary should contain a list of Numpy arrays, giving the data for each
trial within each list in the experiment.
A simple implementation of :code:`prepare_sim` may just call
:py:func:`cymr.fit.prepare_lists`, which converts standard :code:`input`
and :code:`item_index` columns to list format.

All of the information needed to run a simulation or evaluate likelihood
must be included by :code:`prepare_sim`. For example, one model might only
use the serial position of each item, while other models might also take item
identity into account.

Evaluating likelihood
^^^^^^^^^^^^^^^^^^^^^

To support maximum-likelihood estimation of model parameters, the model class
must implement a :code:`likelihood_subject` method that takes in data
for one subject (in Psifr format) and a set of parameter values and returns log
likelihood of the data under the model with those parameters.

Generating simulated data
^^^^^^^^^^^^^^^^^^^^^^^^^

To support generation of simulated data, the model class must implement a
:code:`generate_subject` method that takes information about the experiment to
simulate (in Psifr format) and returns a list of recalled items for each free recall list.
