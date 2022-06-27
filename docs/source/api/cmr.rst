CMR
===

.. currentmodule:: cymr.cmr

Model framework
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    CMR
    CMR.likelihood
    CMR.fit_indiv
    CMR.generate
    CMR.record
    CMR.parameter_sweep
    CMR.parameter_recovery

Model configuration
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    save_patterns
    load_patterns
    read_config
    config_loc_cmr

Model parameters
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    CMRParameters
    CMRParameters.set_fixed
    CMRParameters.set_free
    CMRParameters.set_dependent
    CMRParameters.eval_dependent
    CMRParameters.set_dynamic
    CMRParameters.eval_dynamic
    CMRParameters.set_sublayers
    CMRParameters.set_weights
    CMRParameters.set_sublayer_param
