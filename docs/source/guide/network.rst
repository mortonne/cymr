
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

=====================
Network customization
=====================

Depending on the task you are simulating or the mechanisms
you are testing, you may need to customize the weights in your
network or the parameters applied to different parts of the network.

Network addresses
~~~~~~~~~~~~~~~~~

A network contains multiple subdivisions that help organize how
different parts are used. When initializing a network, you must
indicate which *segments* to include on the item :math:`f` and
:math:`c` layers.

Layers are specified using an address:

layer
    A layer of the network; either :code:`f` or :code:`c`.

sublayer
    A sublayer within that layer. This is used to allow different
    behavior within a layer, for example different context evolution
    rates.

segment
    Segment within the sublayer. This is used for convenience to
    organize sublayers. For example, may designate one segment for
    items, another as a start unit, and another segment to represent
    distractor items.

unit
    Unit within the segment, indicated as an integer. The first unit
    is 0.

These can be specified at different levels of scope. For example, a
segment of a layer is specified using layer, sublayer, and segment labels.

Parts of weight matrices connecting layers are specified using addresses
for both the :math:`f` and :math:`c` layers. An area of a weight matrix
is called a *region* and is indicated by two segment specifications.

.. ipython:: python

    from cymr import network
    f_segment = {'task': {'item': 24, 'start': 1}}
    c_segment = {
        'loc': {'item': 24, 'start': 1},
        'cat': {'item': 3, 'start': 1},
    }
    net = network.Network(f_segment, c_segment)
    print(net)

Setting connection weights
~~~~~~~~~~~~~~~~~~~~~~~~~~

Connection weights may be specified using a patterns dictionary
together with a weights template.

.. ipython:: python

    cat = np.zeros((24, 3))
    cat[:8, 0] = 1
    cat[8:16, 1] = 1
    cat[16:, 2] = 1
    patterns = {
        'vector': {
            'loc': np.eye(24),
            'cat': cat,
        },
    }

Once a set of patterns has been defined, they can be referenced
in a weights template to indicate where they should be placed:

.. ipython:: python

    from cymr import parameters
    param_def = parameters.Parameters()
    weights = {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
    }
    param_def.set_weights('fc', weights)
    param_def.set_weights('cf', weights)
