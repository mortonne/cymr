===
Fit
===

.. currentmodule:: cymr.fit

Utilities
~~~~~~~~~

.. autosummary::
    :toctree: api/

    prepare_lists
    prepare_study
    set_dependent
    get_best_results
    add_recalls
    sample_parameters

Model Evaluation
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Recall.likelihood_subject
    Recall.likelihood

Parameter Sets
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Parameters
    Parameters.add_fixed
    Parameters.add_free
    Parameters.add_dependent
    Parameters.add_weights
    Parameters.to_json

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Recall.fit_subject
    Recall.fit_indiv

Generating Simulated Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Recall.generate_subject
    Recall.generate

Characterizing Model Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Recall.parameter_sweep
    Recall.parameter_recovery
