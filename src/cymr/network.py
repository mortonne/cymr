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
    """Expand a scalar parameter to array format."""
    if not isinstance(param, np.ndarray):
        param = np.tile(param, size).astype(float)
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
        self.f_segment = f_segment
        self.c_segment = c_segment
        self.f_ind = LayerIndex(f_segment)
        self.c_ind = LayerIndex(c_segment)
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
        np.set_printoptions(precision=4, floatmode='fixed', sign=' ')
        s_f = 'f:\n' + self.f.__str__()
        s_c = 'c:\n' + self.c.__str__()
        s_fc_pre = 'W pre [fc]:\n' + self.w_fc_pre.__str__()
        s_fc_exp = 'W exp [fc]:\n' + self.w_fc_exp.__str__()
        s_cf_pre = 'W pre [cf]:\n' + self.w_cf_pre.__str__()
        s_cf_exp = 'W exp [cf]:\n' + self.w_cf_exp.__str__()
        s_ff_pre = 'W pre [ff]:\n' + self.w_ff_pre.__str__()
        s_ff_exp = 'W exp [ff]:\n' + self.w_ff_exp.__str__()
        s = '\n\n'.join([s_f, s_c, s_fc_pre, s_fc_exp, s_cf_pre, s_cf_exp,
                         s_ff_pre, s_ff_exp])
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
        net.f_ind = self.f_ind
        net.c_ind = self.c_ind
        net.n_f = self.n_f
        net.n_c = self.n_c
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
        if layer == 'f':
            ind = self.f_ind.get_sublayer(sublayer)
        elif layer == 'c':
            ind = self.c_ind.get_sublayer(sublayer)
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def get_sublayers(self, layer, sublayers):
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
        f_ind = self.f_ind.get_segment(*f_segment)
        c_ind = self.c_ind.get_segment(*c_segment)
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
        f_ind, c_ind = self.get_region(f_segment, c_segment)
        if connect == 'fc':
            self.w_fc_pre[np.ix_(f_ind, c_ind)] = scaled
        elif connect == 'cf':
            self.w_cf_pre[np.ix_(f_ind, c_ind)] = scaled
        elif connect == 'ff':
            self.w_ff_pre[np.ix_(f_ind, c_ind)] = scaled
        else:
            raise ValueError(f'Invalid connection type: {connect}')

    def update(self, item, sublayer):
        """
        Update context completely with input from the item layer.

        Rather than integrating input into context, this replaces the
        current state of context with the input.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayer : str
            Sublayer of context to update.
        """
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayer('c', sublayer)
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, f_ind, c_ind, B=1)

    def integrate(self, item, sublayer, B):
        """
        Integrate input from the item layer into context.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayer : str
            Sublayer of context to update.

        B : float
            Integration scaling factor; higher values update context
            to more strongly reflect the input.
        """
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayer('c', sublayer)
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, f_ind, c_ind, B)

    def present(self, item, sublayer, B, Lfc=0, Lcf=0):
        """
        Present an item and learn context-item associations.

        Parameters
        ----------
        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayer : str
            Sublayer of context to update.

        B : float
            Integration scaling factor; higher values update context
            to more strongly reflect the input.

        Lfc : float, optional
            Learning rate for item to context associations.

        Lcf : float, optional
            Learning rate for context to item associations.
        """
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayer('c', sublayer)
        operations.present(self.w_fc_exp, self.w_fc_pre, self.w_cf_exp,
                           self.c, self.c_in, self.f, f_ind, c_ind,
                           B, Lfc, Lcf)

    def learn(self, connect, item, sublayer, L):
        """
        Learn an association between the item and context layers.

        Parameters
        ----------
        connect : {'fc', 'cf'}
            Connection matrix to update.

        item : tuple of str, str, int
            Sublayer, segment, and unit of the item to present.

        sublayer : str
            Sublayer of context to update.

        L : double
            Learning rate.
        """
        f_ind = self.get_unit('f', *item)
        c_ind = self.get_sublayer('c', sublayer)
        if connect == 'fc':
            self.w_fc_exp[f_ind, c_ind] += self.c[c_ind] * L
        elif connect == 'cf':
            self.w_cf_exp[f_ind, c_ind] += self.c[c_ind] * L
        else:
            raise ValueError(f'Invalid connection: {connect}')

    def study(self, segment, item_list, sublayer, B, Lfc, Lcf):
        """
        Study a list of items.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment of items to present.

        item_list : numpy.array
            Item indices relative to the segment.

        sublayer : str
            Sublayer of context to update.

        B : float or numpy.array
            Context updating rate. If an array, specifies a rate for
            each individual study trial.

        Lfc : float or numpy.array
            Learning rate for item to context associations. If an
            array, specifies a learning rate for each individual trial.

        Lcf : float or numpy.array
            Learning rate for context to item associations.
        """
        if not isinstance(B, np.ndarray):
            B = np.tile(B, item_list.shape).astype(float)
        if not isinstance(Lfc, np.ndarray):
            Lfc = np.tile(Lfc, item_list.shape).astype(float)
        if not isinstance(Lcf, np.ndarray):
            Lcf = np.tile(Lcf, item_list.shape).astype(float)
        f_ind = self.get_segment('f', *segment)
        item_ind = f_ind[item_list]
        c_ind = self.get_sublayer('c', sublayer)
        operations.study(self.w_fc_exp, self.w_fc_pre,
                         self.w_cf_exp, self.c, self.c_in,
                         self.f, item_ind, c_ind, B, Lfc, Lcf)

    def study_distract(self, segment, item_list, sublayer, B, Lfc, Lcf,
                       distract_segment, distract_list, distract_B):
        """
        Study a list of items.

        Parameters
        ----------
        segment : tuple of str, str
            Sublayer and segment representing the items to be presented.

        item_list : numpy.array
            Item indices relative to the segment.

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
        item_ind = f_ind[item_list]
        if not isinstance(B, np.ndarray):
            B = np.tile(B, item_list.shape).astype(float)
        if not isinstance(Lfc, np.ndarray):
            Lfc = np.tile(Lfc, item_list.shape).astype(float)
        if not isinstance(Lcf, np.ndarray):
            Lcf = np.tile(Lcf, item_list.shape).astype(float)

        f_ind = self.get_segment('f', *distract_segment)
        distract_ind = f_ind[distract_list]

        if not isinstance(distract_B, np.ndarray):
            distract_B = np.tile(
                distract_B, item_list.shape[0] + 1
            ).astype(float)

        c_ind = self.get_sublayer('c', sublayer)
        operations.study_distract(
            self.w_fc_exp, self.w_fc_pre, self.w_cf_exp, self.c, self.c_in,
            self.f, item_ind, c_ind, B, Lfc, Lcf, distract_ind, distract_B
        )

    def record_study(self, segment, item_list, B, Lfc, Lcf,
                     distract_segment=None, distract_list=None,
                     distract_B=None):
        n = len(item_list)
        B = expand_param(B, n)
        Lfc = expand_param(Lfc, n)
        Lcf = expand_param(Lcf, n)
        state = []
        if distract_B is not None:
            distract_B = expand_param(distract_B, n + 1)
        for i in range(len(item_list)):
            if distract_B is not None and distract_B[i] > 0:
                self.integrate(distract_segment, distract_list[i],
                               distract_B[i])
            self.present(segment, item_list[i], B[i], Lfc[i], Lcf[i])
            state.append(self.copy())
        if distract_B is not None and distract_B[n] > 0:
            self.integrate(distract_segment, distract_list[n], distract_B[n])
        state.append(self.copy())
        return state

    def record_recall(self, segment, recalls, B, T, amin=0.000001):
        rec_ind = self.f_ind[segment]
        w_cf = self.w_cf_exp[rec_ind, :] + self.w_cf_pre[rec_ind, :]
        exclude = np.zeros(self.n_f, dtype=bool)
        state = [self.copy()]
        for output, recall in enumerate(recalls):
            # project the current state of context; assume nonzero support
            self.f_in[rec_ind] = np.dot(w_cf, self.c)
            if output > 0:
                item_cue = (self.w_ff_pre[rec_ind, recalls[output - 1]] +
                            self.w_ff_exp[rec_ind, recalls[output - 1]])
                self.f_in[rec_ind] += item_cue
            self.f_in[self.f_in < amin] = amin

            # scale based on choice parameter, set recalled items to zero
            self.f_in[rec_ind] = np.exp((2 * self.f_in[rec_ind]) / T)
            self.f_in[exclude] = 0

            # remove recalled item from competition
            exclude[recall] = True

            # update context
            ind = self.f_ind[segment].start + recall
            self.f[:] = 0
            self.f[ind] = 1
            state.append(self.copy())
            self.present(segment, recall, B)
        state.append(self.copy())
        return state

    def _p_recall_cython(self, segment, recalls, B, T, p_stop, amin=0.000001):
        if not isinstance(B, np.ndarray):
            B = np.tile(B, len(recalls)).astype(float)
        rec_ind = self.f_ind[segment]
        n_item = rec_ind.stop - rec_ind.start
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        p = np.zeros(len(recalls) + 1)
        recalls = np.array(recalls, dtype=np.dtype('i'))
        operations.p_recall(rec_ind.start, n_item, recalls,
                            self.w_fc_exp, self.w_fc_pre,
                            self.w_cf_exp, self.w_cf_pre,
                            self.w_ff_exp, self.w_ff_pre,
                            self.f, self.f_in, self.c, self.c_in,
                            exclude, amin, B, T, p_stop, p)
        return p

    def _p_recall_python(self, segment, recalls, B, T, p_stop, amin=0.000001):
        # weights to use for recall (assume fixed during recall)
        rec_ind = self.f_ind[segment]
        w_cf = self.w_cf_exp[rec_ind, :] + self.w_cf_pre[rec_ind, :]

        exclude = np.zeros(w_cf.shape[0], dtype=bool)
        p = np.zeros(len(recalls) + 1)
        for output, recall in enumerate(recalls):
            # project the current state of context; assume nonzero support
            support = np.dot(w_cf, self.c)
            support[support < amin] = amin

            # scale based on choice parameter, set recalled items to zero
            strength = np.exp((2 * support) / T)
            strength[exclude] = 0

            # probability of this recall, conditional on not stopping
            p[output] = ((strength[recall] / np.sum(strength)) *
                         (1 - p_stop[output]))

            # remove recalled item from competition
            exclude[recall] = True

            # update context
            self.present(segment, recall, B)

        # probability of the stopping event
        p[len(recalls)] = p_stop[len(recalls)]
        return p

    def p_recall(self, segment, recalls, B, T, p_stop, amin=0.000001,
                 compiled=True):
        """
        Calculate the probability of a specific recall sequence.

        Parameters
        ----------
        segment : str
            Segment from which items are recalled.

        recalls : numpy.array
            Index of each recalled item relative to the segment.

        B : float
            Context updating rate after each recalled item.

        T : float
            Decision parameter for choice rule.

        p_stop : numpy.array
            Probability of stopping recall at each output position.

        amin : float, optional
            Minimum activation for each not-yet-recalled item on each
            recall attempt.

        compiled : bool, optional
            If true, the compiled version of the function will be used.
        """
        if T < amin:
            T = amin
        if compiled:
            p = self._p_recall_cython(segment, recalls, B, T, p_stop, amin)
        else:
            p = self._p_recall_python(segment, recalls, B, T, p_stop, amin)
        return p

    def generate_recall(self, segment, B, T, p_stop, amin=0.000001):
        """
        Generate a sequence of simulated free recall events.

        Parameters
        ----------
        segment : str
            Segment to retrieve items from.

        B : float
            Context updating rate after each recall.

        T : float
            Decision parameter for choice rule.

        p_stop : numpy.array
            Probability of stopping at each output position.

        amin : float, optional
            Minimum activation of each not-yet-recalled item on each
            recall attempt.
        """
        # weights to use for recall (assume fixed during recall)
        rec_ind = self.f_ind[segment]
        n_item = self.n_f_segment[segment]
        if not isinstance(B, np.ndarray):
            B = np.tile(B, n_item).astype(float)
        if not isinstance(T, np.ndarray):
            T = np.tile(T, n_item).astype(float)

        recalls = []
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        item_ind = np.arange(n_item)
        for i in range(n_item):
            # stop recall with some probability
            if np.random.rand() < p_stop[i]:
                break

            # calculate item support
            operations.cue_item(
                rec_ind.start, n_item, self.w_cf_pre, self.w_cf_exp,
                self.w_ff_pre, self.w_ff_exp, self.f_in, self.c, exclude,
                np.asarray(recalls, dtype=np.dtype('i')), i
            )
            operations.apply_softmax(rec_ind.start, n_item, self.f_in,
                                     exclude, amin, T[i])

            # select item for recall proportionate to support
            support = self.f_in[rec_ind]
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
            self.integrate(segment, recall, B[i])
        return recalls

    def generate_recall_lba(self, segment, time_limit, B, A, b, s, tau):
        """Generate timed free recall using the LBA model."""
        rec_ind = self.f_ind[segment]
        n_item = self.n_f_segment[segment]

        recalls = []
        times = []
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        t = 0
        for i in range(n_item):
            if t >= time_limit:
                break

            # calculate item support
            operations.cue_item(
                rec_ind.start, n_item, self.w_cf_pre, self.w_cf_exp,
                self.w_ff_pre, self.w_ff_exp, self.f_in, self.c, exclude,
                np.asarray(recalls, dtype=np.dtype('i')), i
            )
            support = self.f_in[rec_ind]

            # simulate response competition
            recall, irt = sample_response_lba(A, b, support, s, tau)
            t += irt

            if t <= time_limit:
                # if response happened in time, count it
                recalls.append(recall)
                times.append(t)
                exclude[recall] = 1

                # integrate context associated with the item into context
                self.integrate(segment, recall, B)
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
