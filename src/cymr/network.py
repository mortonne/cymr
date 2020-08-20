"""Represent interactions between context and item layers."""

import numpy as np
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cymr import operations


def save_patterns(h5_file, items, **kwargs):
    """
    Write patterns and similarity matrices to hdf5.

    Parameters
    ----------
    h5_file : str
        Path to hdf5 file to save patterns in.

    items : list of str
        Item strings corresponding to the patterns.

    Additional keyword arguments set named feature vectors. Feature
    vector arrays must have shape [items x units].
    """
    with h5py.File(h5_file, 'w') as f:
        # items
        dt = h5py.special_dtype(vlen=str)
        items = np.asarray(items)
        dset = f.create_dataset('items', items.shape, dtype=dt)
        for i, item in enumerate(items):
            dset[i] = item

        # features
        features = list(kwargs.keys())
        dset = f.create_dataset('features', (len(features),), dtype=dt)
        for i, feature in enumerate(features):
            dset[i] = feature

        # patterns
        for name, vectors in kwargs.items():
            # save vectors
            f.create_dataset('vector/' + name, data=vectors)

            # set pattern similarity to dot product
            sim = np.dot(vectors, vectors.T)
            f.create_dataset('similarity/' + name, data=sim)


def load_patterns(h5_file, features=None):
    """
    Load weights from an hdf5 file.

    Parameters
    ----------
    h5_file : str
        Path to file saved with `save_patterns`.

    features : list of str, optional
        Names of features to load. Default is to load all features.

    Returns
    -------
    patterns : dict of (str: dict of (str: numpy.array))
        Loaded patterns. The "vector" field contains vector patterns.
        The "similarity" field contains pairwise similarity matrices.
        Each type of pattern contains a field for each loaded feature.
    """
    with h5py.File(h5_file, 'r') as f:
        patterns = {'items': f['items'][()],
                    'vector': {},
                    'similarity': {}}

        if features is None:
            features = f['features'][()]

        for name in features:
            patterns['vector'][name] = f['vector/' + name][()]
            patterns['similarity'][name] = f['similarity/' + name][()]
    return patterns


def prepare_patterns(patterns, weights):
    """Scale and concatenate item patterns and connections."""
    scaled = {'fcf': None, 'ff': None}
    if 'fcf' in weights:
        # scale weights
        w = np.array(list(weights['fcf'].values()))
        ws = w / np.linalg.norm(w, ord=2)
        features = list(weights['fcf'].keys())

        # apply weights and make full patterns
        fcf = []
        for feature, wr in zip(features, ws):
            mat = patterns['vector'][feature] * wr
            fcf.append(mat)
        scaled['fcf'] = np.hstack(fcf)

    if 'ff' in weights:
        w = np.array(list(weights['ff'].values()))
        ws = w / np.linalg.norm(w, ord=1)
        features = list(weights['ff'].keys())

        # sum weights
        w_shape = patterns['similarity'][features[0]].shape
        scaled['ff'] = np.zeros(w_shape)
        for feature, wr in zip(features, ws):
            mat = patterns['similarity'][feature] * wr
            scaled['ff'] += mat
    return scaled


def unpack_weights(weight_template, weight_param):
    """Apply parameter values to a weight template."""
    weights = {f: {k: weight_param[v] for k, v in w.items()}
               for f, w in weight_template.items()}
    return weights


def expand_param(param, size):
    """
    Expand a scalar parameter to array format.

    Parameters
    ----------
    param : float or numpy.ndarray
        Parameter to expand.

    size : iterable or numpy.ndarray
        Size of the expanded parameter.

    Returns
    -------
    param : numpy.ndarray
        Expanded parameter.
    """
    size = np.asarray(size)
    if not isinstance(param, np.ndarray):
        # expand scalar to full array
        param = np.tile(param, size).astype(float)
    elif size.shape and param.ndim < len(size):
        # expand array to have to correct number of dimensions
        axis = tuple(size[param.ndim:])
        param = np.expand_dims(param, axis)
        if param.shape != tuple(size):
            # expand singleton dimensions as needed
            rep = np.ones(size.shape)
            for i, n in enumerate(param.shape):
                if n == 1:
                    rep[i] = size[i]
                elif n != size[i] and n < size[i]:
                    raise ValueError('Cannot expand parameter.')
            param = np.tile(param, rep).astype(float)
    return param


def sample_response_lba(A, b, v, s, tau):
    """Sample a response and response time."""
    while True:
        k = stats.uniform.rvs(0, scale=A)
        d = stats.norm.rvs(loc=v, scale=s)
        t = tau + (b - k) / d
        if np.any(d > 0):
            break
    t[d <= 0] = np.nan
    response = np.nanargmin(t)
    rt = np.nanmin(t)
    return int(response), rt


def init_plot(**kwargs):
    fig = plt.figure(constrained_layout=True, **kwargs)
    gs = GridSpec(10, 5, figure=fig)
    ax = {'c': fig.add_subplot(gs[0, 1:4]),
          'c_in': fig.add_subplot(gs[1, 1:4]),
          'f_in': fig.add_subplot(gs[8, 1:4]),
          'f': fig.add_subplot(gs[9, 1:4]),
          'w_fc_pre': fig.add_subplot(gs[2:8, 0]),
          'w_fc_exp': fig.add_subplot(gs[2:8, 1]),
          'w_ff_pre': fig.add_subplot(gs[2:8, 2]),
          'w_cf_exp': fig.add_subplot(gs[2:8, 3]),
          'w_cf_pre': fig.add_subplot(gs[2:8, 4])}
    return fig, ax


class LayerIndex(object):
    def __init__(self, layer_segments):
        self.size = 0
        self.size_sublayer = {}
        self.size_segment = layer_segments.copy()
        self.sublayer = {}
        self.segment = {}
        for sub, segs in layer_segments.items():
            self.segment[sub] = {}
            self.size_sublayer[sub] = 0
            start = self.size
            for seg, s in segs.items():
                self.segment[sub][seg] = np.array(
                    [self.size, self.size + s], dtype=np.dtype('i')
                )
                self.size += s
                self.size_sublayer[sub] += s
            self.sublayer[sub] = np.array(
                [start, start + self.size_sublayer[sub]], dtype=np.dtype('i')
            )

    def __repr__(self):
        s = ''
        for sublayer, segments in self.size_segment.items():
            size_sublayer = self.size_sublayer[sublayer]
            s += f'{sublayer}: {size_sublayer} units\n'
            for segment, size_segment in segments.items():
                s += f'    {segment}: {size_segment} units\n'
        return s

    def copy(self):
        return self.__init__(self.size_segment)

    def get_sublayer(self, sublayer):
        return self.sublayer[sublayer]

    def get_segment(self, sublayer, segment):
        return self.segment[sublayer][segment]

    def get_unit(self, sublayer, segment, index):
        return self.segment[sublayer][segment][0] + index


class Network(object):
    """
    Representation of interacting item and context layers.

    Parameters
    ----------
    f_segment : dict of str: (dict of str: int)
        For each item sublayer, the number of units in each segment.

    c_segment : dict of str: (dict of str: int)
        For each context sublayer, the number of units in each segment.

    Attributes
    ----------
    f_segment : dict of str: (dict of str: int)
        Number of item units for each segment.

    c_segment : dict of str: (dict of str: int)
        Number of context units for each segment.

    f_ind : cymr.network.LayerIndex
        Index of units in the item layer.

    c_ind : cymr.network.LayerIndex
        Index of units in the context layer.

    n_f : int
        Total number of item units.

    n_c : int
        Total number of context units.

    f : numpy.array
        Item layer vector.

    c : numpy.array
        Context layer vector.

    c_in : numpy.array
        Current input to context.

    w_fc_pre : numpy.array
        Pre-experimental weights connecting f to c.

    w_fc_exp : numpy.array
        Weights learned during the experiment connecting f to c.

    w_cf_pre : numpy.array
        Pre-experimental weights connecting c to f.

    w_cf_exp : numpy.array
        Weights learned during the experiment connect c to f.
    w_ff_pre : numpy.array
        Pre-experimental weights connecting f to f.

    w_ff_exp : numpy.array
        Weights learned during the experiment connecting f to f.
    """
    def __init__(self, f_segment, c_segment):
        self.f_segment = f_segment.copy()
        self.c_segment = c_segment.copy()
        self.f_sublayers = list(self.f_segment.keys())
        self.c_sublayers = list(self.c_segment.keys())
        self.f_ind = LayerIndex(self.f_segment)
        self.c_ind = LayerIndex(self.c_segment)
        n_f = self.f_ind.size
        n_c = self.c_ind.size
        self.n_f = n_f
        self.n_c = n_c
        self.f = np.zeros(n_f)
        self.f_in = np.zeros(n_f)
        self.c = np.zeros(n_c)
        self.c_in = np.zeros(n_c)
        self.w_fc_pre = np.zeros((n_f, n_c))
        self.w_fc_exp = np.zeros((n_f, n_c))
        self.w_cf_pre = np.zeros((n_f, n_c))
        self.w_cf_exp = np.zeros((n_f, n_c))
        self.w_ff_pre = np.zeros((n_f, n_f))
        self.w_ff_exp = np.zeros((n_f, n_f))

    def __repr__(self):
        s = f'f:\n{self.f_ind}\nc:\n{self.c_ind}'
        return s

    def reset(self):
        """Reset network weights and activations to zero."""
        self.f[:] = 0
        self.c[:] = 0
        self.c_in[:] = 0
        self.w_fc_exp[:] = 0
        self.w_fc_pre[:] = 0
        self.w_cf_exp[:] = 0
        self.w_cf_pre[:] = 0
        self.w_ff_exp[:] = 0
        self.w_ff_pre[:] = 0

    def copy(self):
        """
        Copy the network to a new network object.

        Returns
        -------
        net : cymr.Network
            Network with the same segments, weights, and activations.
        """
        net = Network(self.f_segment, self.c_segment)
        net.f = self.f.copy()
        net.f_in = self.f_in.copy()
        net.c = self.c.copy()
        net.c_in = self.c_in.copy()
        net.w_fc_exp = self.w_fc_exp.copy()
        net.w_fc_pre = self.w_fc_pre.copy()
        net.w_cf_exp = self.w_cf_exp.copy()
        net.w_cf_pre = self.w_cf_pre.copy()
        net.w_ff_exp = self.w_ff_exp.copy()
        net.w_ff_pre = self.w_ff_pre.copy()
        return net

    def get_sublayer(self, layer, sublayer):
        """Get indices for a sublayer."""
        if layer == 'f':
            ind = self.f_ind.get_sublayer(sublayer)
        elif layer == 'c':
            ind = self.c_ind.get_sublayer(sublayer)
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def get_sublayers(self, layer, sublayers):
        """Get an array of indices for multiple sublayers."""
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        ind_list = []
        for sublayer in sublayers:
            ind_list.append(self.get_sublayer(layer, sublayer))
        return np.array(ind_list, dtype=np.dtype('i'))

    def get_region(self, f_segment, c_segment):
        """
        Return slices for a region.

        Returns
        -------
        f_ind : slice
            Span of the region in the item dimension.

        c_ind : slice
            Span of the region in the context dimension.
        """
        f_ind = slice(*tuple(self.f_ind.get_segment(*f_segment)))
        c_ind = slice(*tuple(self.c_ind.get_segment(*c_segment)))
        return f_ind, c_ind

    def get_segment(self, layer, sublayer, segment):
        """
        Get indices for a segment.

        Parameters
        ----------
        layer : {'f', 'c'}
            Layer to access.

        sublayer : str
            Sublayer to access.

        segment : str
            Segment to access.

        Returns
        -------
        ind : numpy.array
            Segment indices.
        """
        if layer == 'f':
            ind = self.f_ind.get_segment(sublayer, segment)
        elif layer == 'c':
            ind = self.c_ind.get_segment(sublayer, segment)
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def get_unit(self, layer, sublayer, segment, unit):
        """
        Get indices for a unit.

        Parameters
        ----------
        layer : {'f', 'c'}
            Layer to access.

        sublayer : str
            Sublayer to access.

        segment : str
            Segment to access.

        unit : int
            Index of the unit within the segment.

        Returns
        -------
        ind : int
            Absolute index of the unit within the layer.
        """
        if layer == 'f':
            ind = self.f_ind.get_unit(sublayer, segment, unit)
        elif layer == 'c':
            ind = self.c_ind.get_unit(sublayer, segment, unit)
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def add_pre_weights(self, connect, f_segment, c_segment, weights,
                        slope=1, intercept=0):
        """
        Add pre-experimental weights to a network.

        Parameters
        ----------
        connect : {'fc', 'cf'}
            Connections to add weights to.

        f_segment : tuple of str, str
            Sublayer and segment to use for the item layer.

        c_segment : tuple of str, str
            Sublayer and segment to use for the context layer.

        weights : numpy.array
            Items x context array of weights.

        slope : double, optional
            Slope to multiply weights by before adding.

        intercept : double, optional
            Intercept to add to weights.
        """
        scaled = intercept + slope * weights
        if connect == 'ff':
            f_slice = slice(*tuple(self.f_ind.get_segment(*f_segment)))
            c_slice = None
        else:
            f_slice, c_slice = self.get_region(f_segment, c_segment)
        if connect == 'fc':
            self.w_fc_pre[f_slice, c_slice] = scaled
        elif connect == 'cf':
            self.w_cf_pre[f_slice, c_slice] = scaled
        elif connect == 'ff':
            self.w_ff_pre[f_slice, f_slice] = scaled
        else:
            raise ValueError(f'Invalid connection type: {connect}')

    def update(self, item, sublayers):
        """
        Update context completely with input from the item layer.

        Rather than integrating input into context, this replaces the
        current state of context with the input.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayers : str of list of str
            Sublayer(s) of context to update.
        """
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayers('c', sublayers)
        B = np.ones(len(sublayers))
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, f_ind, c_ind, B)

    def integrate(self, item, sublayers, B):
        """
        Integrate input from the item layer into context.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayers : str or list of str
            Sublayer(s) of context to update.

        B : float or numpy.array
            Integration scaling factor for each sublayer; higher values
            update context to more strongly reflect the input.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        B = expand_param(B, np.array(len(sublayers)))
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayers('c', sublayers)
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, f_ind, c_ind, B)

    def present(self, item, sublayers, B, Lfc=0, Lcf=0):
        """
        Present an item and learn context-item associations.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayers : str or list of str
            Sublayer(s) of context to update.

        B : float or numpy.ndarray
            Integration scaling factors; higher values update context
            to more strongly reflect the input.

        Lfc : float or numpy.ndarray, optional
            Learning rates for item to context associations.

        Lcf : float or numpy.ndarray, optional
            Learning rates for context to item associations.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        n_sub = np.array(len(sublayers))
        B = expand_param(B, n_sub)
        Lfc = expand_param(Lfc, n_sub)
        Lcf = expand_param(Lcf, n_sub)
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayers('c', sublayers)
        operations.present(self.w_fc_exp, self.w_fc_pre, self.w_cf_exp,
                           self.c, self.c_in, self.f, f_ind, c_ind,
                           B, Lfc, Lcf)

    def learn(self, connect, item, sublayers, L):
        """
        Learn an association between the item and context layers.

        Parameters
        ----------
        connect : {'fc', 'cf'}
            Connection matrix to update.

        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayers : str
            Sublayers of context to update.

        L : double
            Learning rate.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        L = expand_param(L, len(sublayers))
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayers('c', sublayers)
        if connect == 'fc':
            mat = self.w_fc_exp
        elif connect == 'cf':
            mat = self.w_cf_exp
        else:
            raise ValueError(f'Invalid connection: {connect}')
        for i, s_ind in enumerate(c_ind):
            s_slice = slice(*tuple(s_ind))
            mat[f_ind, s_slice] += self.c[s_slice] * L[i]

    def study(self, segment, item_list, sublayers, B, Lfc, Lcf):
        """
        Study a list of items.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment of items to present.

        item_list : numpy.array
            Item indices relative to the segment.

        sublayers : str or list of str
            Sublayers of context to update.

        B : float or numpy.array
            Context updating rate. If an array, specifies a rate for
            each individual study trial.

        Lfc : float or numpy.array
            Learning rate for item to context associations. If an
            array, specifies a learning rate for each individual trial.

        Lcf : float or numpy.array
            Learning rate for context to item associations.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        n_item = item_list.shape[0]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))
        Lfc = expand_param(Lfc, (n_item, n_sub))
        Lcf = expand_param(Lcf, (n_item, n_sub))
        f_ind = self.get_segment('f', *segment)
        item_ind = (f_ind[0] + item_list).astype(np.dtype('i'))
        c_ind = self.get_sublayers('c', sublayers)
        operations.study(self.w_fc_exp, self.w_fc_pre,
                         self.w_cf_exp, self.c, self.c_in,
                         self.f, item_ind, c_ind, B, Lfc, Lcf)

    def study_distract(self, segment, item_list, sublayers, B, Lfc, Lcf,
                       distract_segment, distract_list, distract_B):
        """
        Study a list of items.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment representing the items to be presented.

        item_list : numpy.array
            Item indices relative to the segment.

        sublayers : str or list of str
            Sublayers to update during study.

        B : float or numpy.array
            Context updating rate. If an array, specifies a rate for
            each individual study trial.

        Lfc : float or numpy.array
            Learning rate for item to context associations. If an
            array, specifies a learning rate for each individual trial.

        Lcf : float or numpy.array
            Learning rate for context to item associations.

        distract_segment : str, str
            Sublayer and segment representing distraction trials.

        distract_list : numpy.array
            Distraction item indices relative to the segment.

        distract_B : float or numpy.array
            Context updating rate for each distraction event before
            and after each study event. If an array, must be of length
            n_items + 1. Distraction will not be presented on trials i
            where distract_B[i] is zero.
        """
        f_ind = self.get_segment('f', *segment)
        item_ind = (f_ind[0] + item_list).astype(np.dtype('i'))
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        n_item = item_list.shape[0]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))
        Lfc = expand_param(Lfc, (n_item, n_sub))
        Lcf = expand_param(Lcf, (n_item, n_sub))

        f_ind = self.get_segment('f', *distract_segment)
        distract_ind = (f_ind[0] + distract_list).astype(np.dtype('i'))

        if not isinstance(distract_B, np.ndarray):
            distract_B = np.tile(distract_B, (n_item + 1, n_sub)).astype(float)

        c_ind = self.get_sublayers('c', sublayers)
        operations.study_distract(
            self.w_fc_exp, self.w_fc_pre, self.w_cf_exp, self.c, self.c_in,
            self.f, item_ind, c_ind, B, Lfc, Lcf, distract_ind, distract_B
        )

    def record_study(self, segment, item_list, sublayers, B, Lfc, Lcf):
        n_item = len(item_list)
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))
        Lfc = expand_param(Lfc, (n_item, n_sub))
        Lcf = expand_param(Lcf, (n_item, n_sub))
        state = []
        for i in range(len(item_list)):
            item = (*segment, item_list[i])
            self.present(item, sublayers, B[i], Lfc[i], Lcf[i])
            state.append(self.copy())
        state.append(self.copy())
        return state

    def record_recall(self, segment, recalls, sublayers, B, T, amin=0.000001):
        rec_ind = self.get_segment('f', *segment)
        rec_slice = slice(rec_ind[0], rec_ind[1])
        w_cf = self.w_cf_exp[rec_slice, :] + self.w_cf_pre[rec_slice, :]
        exclude = np.zeros(self.n_f, dtype=bool)
        state = [self.copy()]
        for output, recall in enumerate(recalls):
            # project the current state of context; assume nonzero support
            self.f_in[rec_slice] = np.dot(w_cf, self.c)
            if output > 0:
                item_cue = (self.w_ff_pre[rec_slice, recalls[output - 1]] +
                            self.w_ff_exp[rec_slice, recalls[output - 1]])
                self.f_in[rec_slice] += item_cue
            self.f_in[self.f_in < amin] = amin

            # scale based on choice parameter, set recalled items to zero
            self.f_in[rec_slice] = np.exp((2 * self.f_in[rec_slice]) / T)
            self.f_in[exclude] = 0

            # remove recalled item from competition
            exclude[recall] = True

            # update context
            ind = rec_ind[0] + recall
            self.f[:] = 0
            self.f[ind] = 1
            state.append(self.copy())
            item = (*segment, recall)
            self.present(item, sublayers, B)
        state.append(self.copy())
        return state

    def p_recall(self, segment, recalls, sublayers, B, T, p_stop, amin=0.000001):
        """
        Calculate the probability of a specific recall sequence.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment from which items are recalled.

        recalls : numpy.array
            Index of each recalled item relative to the segment.

        sublayers : str or list of str
            Sublayer(s) of context to update.

        B : float
            Context updating rate after each recalled item.

        T : float
            Decision parameter for choice rule.

        p_stop : numpy.array
            Probability of stopping recall at each output position.

        amin : float, optional
            Minimum activation for each not-yet-recalled item on each
            recall attempt.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        recalls = np.array(recalls, dtype=np.dtype('i'))
        n_item = recalls.shape[0]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))
        if T < amin:
            T = amin
        f_ind = self.get_segment('f', *segment)
        start = f_ind[0]
        n_f = f_ind[1] - f_ind[0]
        c_ind = self.get_sublayers('c', sublayers)

        exclude = np.zeros(n_f, dtype=np.dtype('i'))
        p = np.zeros(len(recalls) + 1)
        operations.p_recall(
            start, n_f, recalls, self.w_fc_exp, self.w_fc_pre,
            self.w_cf_exp, self.w_cf_pre, self.w_ff_exp, self.w_ff_pre,
            self.f, self.f_in, self.c, self.c_in, c_ind, exclude,
            amin, B, T, p_stop, p
        )
        return p

    def generate_recall(self, segment, sublayers, B, T, p_stop, amin=0.000001):
        """
        Generate a sequence of simulated free recall events.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment to retrieve items from.

        sublayers : str or list of str
            Sublayer(s) of context to update.

        B : float or numpy.ndarray
            Context updating rate after each recall.

        T : float
            Decision parameter for choice rule.

        p_stop : numpy.array
            Probability of stopping at each output position.

        amin : float, optional
            Minimum activation of each not-yet-recalled item on each
            recall attempt.
        """
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        # weights to use for recall (assume fixed during recall)
        rec_ind = self.get_segment('f', *segment)
        n_item = rec_ind[1] - rec_ind[0]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))

        recalls = []
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        item_ind = np.arange(n_item)
        for i in range(n_item):
            # stop recall with some probability
            if np.random.rand() < p_stop[i]:
                break

            # calculate item support
            operations.cue_item(
                rec_ind[0], n_item, self.w_cf_pre, self.w_cf_exp,
                self.w_ff_pre, self.w_ff_exp, self.f_in, self.c, exclude,
                np.asarray(recalls, dtype=np.dtype('i')), i
            )
            operations.apply_softmax(rec_ind[0], n_item, self.f_in,
                                     exclude, amin, T)

            # select item for recall proportionate to support
            support = self.f_in[rec_ind[0]:rec_ind[1]]
            p_recall = support / np.sum(support)
            if np.any(np.isnan(p_recall)):
                n = np.count_nonzero(exclude == 0)
                p_recall[exclude == 0] = 1 / n
                p_recall[exclude == 1] = 0
                recall = np.random.choice(item_ind, p=p_recall)
            else:
                recall = np.random.choice(item_ind, p=p_recall)
            recalls.append(recall)
            exclude[recall] = 1

            # integrate context associated with the item into context
            item = (*segment, recall)
            self.integrate(item, sublayers, B[i])
        return recalls

    def generate_recall_lba(self, segment, sublayers, time_limit, B, A, b, s, tau):
        """Generate timed free recall using the LBA model."""
        if not isinstance(sublayers, list):
            sublayers = [sublayers]
        rec_ind = self.get_segment('f', *segment)
        n_item = rec_ind[1] - rec_ind[0]
        n_sub = len(sublayers)
        B = expand_param(B, (n_item, n_sub))

        recalls = []
        times = []
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        t = 0
        for i in range(n_item):
            if t >= time_limit:
                break

            # calculate item support
            operations.cue_item(
                rec_ind[0], n_item, self.w_cf_pre, self.w_cf_exp,
                self.w_ff_pre, self.w_ff_exp, self.f_in, self.c, exclude,
                np.asarray(recalls, dtype=np.dtype('i')), i
            )
            support = self.f_in[rec_ind[0]:rec_ind[1]]

            # simulate response competition
            recall, irt = sample_response_lba(A, b, support, s, tau)
            t += irt

            if t <= time_limit:
                # if response happened in time, count it
                recalls.append(recall)
                times.append(t)
                exclude[recall] = 1

                # integrate context associated with the item into context
                item = (*segment, recall)
                self.integrate(item, sublayers, B[i])
        return recalls, times

    def plot(self, ax=None):
        """Plot the current state of the network."""
        if ax is None:
            ax = init_plot()
        for name, h in ax.items():
            if not hasattr(self, name):
                continue
            mat = getattr(self, name)
            if mat.ndim == 1:
                mat = mat[None, :]
            h.matshow(mat)
            h.set_aspect('auto')
            h.set_axis_off()
            h.set_title(name, fontsize=12)
