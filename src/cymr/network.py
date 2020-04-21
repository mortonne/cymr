"""Represent interactions between context and item layers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cymr import operations


def primacy(n_item, L, P1, P2):
    """
    Primacy gradient in learning rate.

    Parameters
    ----------
    n_item : int
        Number of items in study list.

    L : float
        Base learning rate. Asymptote of gradient for later positions.

    P1 : float
        Additional learning for first item.

    P2 : float
        Decay rate for primacy gradient.

    Returns
    -------
    rate : numpy.array
        Learning rate for each serial position.
    """
    position = np.arange(n_item)
    rate = L + (P1 * np.exp(-P2 * position))
    return rate


def p_stop_op(n_item, X1, X2, pmin=0.000001):
    """
    Probability of stopping based on output position.

    Parameters
    ----------
    n_item : int
        Number of items available for recall.

    X1 : float
        Probability of not recalling any items.

    X2 : float
        Shape parameter of exponential function increasing stop
        probability by output position.

    pmin : float, optional
        Minimum probability of stopping recall at any position.
    """
    p_stop = X1 * np.exp(X2 * np.arange(n_item + 1))
    p_stop[p_stop < pmin] = pmin
    p_stop[p_stop > (1 - pmin)] = 1 - pmin

    # after recalling all items, P(stop)=1 by definition
    p_stop[-1] = 1
    return p_stop


class Network(object):
    """
    Representation of interacting item and context layers.

    Parameters
    ----------
    segments : dict of {str: (int, int)}
        Definition of network segments. For example, may have a segment
        representing learned items and a segment representing
        distraction trials. Each entry contains an (n_f, n_c) pair
        indicating the number of item and context units to allocate for
        that segment.

    Attributes
    ----------
    segments : dict of {str: (int, int)}
        Number of item and context units for each named segment.

    n_f_segment : dict of {str: int}
        Number of item units for each segment.

    n_c_segment : dict of {str: int}
        Number of context units for each segment.

    f_ind : dict of {str: slice}
        Slice object for item units for each segment.

    c_ind : dict of {str: slice}
        Slice object for context units for each segment

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
    """
    def __init__(self, segments):
        n_f = 0
        n_c = 0
        self.segments = segments
        self.n_f_segment = {}
        self.n_c_segment = {}
        self.f_ind = {}
        self.c_ind = {}
        for name, (s_f, s_c) in segments.items():
            self.n_f_segment[name] = s_f
            self.n_c_segment[name] = s_c
            self.f_ind[name] = slice(n_f, n_f + s_f)
            self.c_ind[name] = slice(n_c, n_c + s_c)
            n_f += s_f
            n_c += s_c
        self.f_ind['all'] = slice(0, n_f)
        self.c_ind['all'] = slice(0, n_c)

        self.n_f = n_f
        self.n_c = n_c
        self.f = np.zeros(n_f)
        self.c = np.zeros(n_c)
        self.c_in = np.zeros(n_c)
        self.w_fc_pre = np.zeros((n_f, n_c))
        self.w_fc_exp = np.zeros((n_f, n_c))
        self.w_cf_pre = np.zeros((n_f, n_c))
        self.w_cf_exp = np.zeros((n_f, n_c))

    def __repr__(self):
        np.set_printoptions(precision=4, floatmode='fixed', sign=' ')
        s_f = 'f:\n' + self.f.__str__()
        s_c = 'c:\n' + self.c.__str__()
        s_fc_pre = 'W pre [fc]:\n' + self.w_fc_pre.__str__()
        s_fc_exp = 'W exp [fc]:\n' + self.w_fc_exp.__str__()
        s_cf_pre = 'W pre [cf]:\n' + self.w_cf_pre.__str__()
        s_cf_exp = 'W exp [cf]:\n' + self.w_cf_exp.__str__()
        s = '\n\n'.join([s_f, s_c, s_fc_pre, s_fc_exp, s_cf_pre, s_cf_exp])
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

    def copy(self):
        """
        Copy the network to a new network object.

        Returns
        -------
        net : cymr.Network
            Network with the same segments, weights, and activations.
        """
        net = Network(self.segments)
        net.f_ind = self.f_ind
        net.c_ind = self.c_ind
        net.n_f = self.n_f
        net.n_c = self.n_c
        net.f = self.f.copy()
        net.c = self.c.copy()
        net.c_in = self.c_in.copy()
        net.w_fc_exp = self.w_fc_exp.copy()
        net.w_fc_pre = self.w_fc_pre.copy()
        net.w_cf_exp = self.w_cf_exp.copy()
        net.w_cf_pre = self.w_cf_pre.copy()
        return net

    def get_slices(self, region):
        """
        Return slices for a region.

        Returns
        -------
        f_ind : slice
            Span of the region in the item dimension.

        c_ind : slice
            Span of the region in the context dimension.
        """
        f_ind = self.f_ind[region[0]]
        c_ind = self.c_ind[region[1]]
        return f_ind, c_ind

    def get_ind(self, layer, segment, item):
        """
        Get the absolute index for an item.

        Parameters
        ----------
        layer : {'f', 'c'}
            Layer to access.

        segment : str
            Segment to access.

        item : int
            Index relative to the start of the segment.

        Returns
        -------
        ind : int
            Absolute index.
        """
        if layer == 'f':
            ind = self.f_ind[segment].start + item
        elif layer == 'c':
            ind = self.c_ind[segment].start + item
        else:
            raise ValueError(f'Invalid layer: {layer}')
        return ind

    def add_pre_weights(self, connect, region, weights, slope=1, intercept=0):
        """
        Add pre-experimental weights to a network.

        Parameters
        ----------
        connect : {'fc', 'cf'}
            Connections to add weights to.

        region : tuple of str, str
            Combination of segments to add the weights to.

        weights : numpy.array
            Items x context array of weights.

        slope : double, optional
            Slope to multiply weights by before adding.

        intercept : double, optional
            Intercept to add to weights.
        """
        scaled = intercept + slope * weights
        f_ind, c_ind = self.get_slices(region)
        if connect == 'fc':
            self.w_fc_pre[f_ind, c_ind] = scaled
        elif connect == 'cf':
            self.w_cf_pre[f_ind, c_ind] = scaled
        else:
            raise ValueError(f'Invalid connection type: {connect}')

    def update(self, segment, item):
        """
        Update context completely with input from the item layer.

        Rather than integrating input into context, this replaces the
        current state of context with the input.

        Parameters
        ----------
        segment : str
            Segment of the item layer to cue with.

        item : int
            Item index within the segment to present.
        """
        ind = self.f_ind[segment].start + item
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, ind, B=1)

    def integrate(self, segment, item, B):
        """
        Integrate input from the item layer into context.

        Parameters
        ----------
        segment : str
            Segment of the item layer to cue with.

        item : int
            Item index within the segment to present.

        B : float
            Integration scaling factor; higher values update context
            to more strongly reflect the input.
        """
        ind = self.f_ind[segment].start + item
        operations.integrate(self.w_fc_exp, self.w_fc_pre, self.c, self.c_in,
                             self.f, ind, B)

    def present(self, segment, item, B, Lfc=0, Lcf=0):
        """
        Present an item and learn context-item associations.

        Parameters
        ----------
        segment : str
            Segment of the item layer to cue with.

        item : int
            Item index within the segment to present.

        B : float
            Integration scaling factor; higher values update context
            to more strongly reflect the input.

        Lfc : float, optional
            Learning rate for item to context associations.

        Lcf : float, optional
            Learning rate for context to item associations.
        """
        ind = self.f_ind[segment].start + item
        operations.present(self.w_fc_exp, self.w_fc_pre,
                           self.w_cf_exp,
                           self.c, self.c_in, self.f, ind, B,
                           Lfc, Lcf)

    def learn(self, connect, segment, item, L):
        """
        Learn an association between the item and context layers.

        Parameters
        ----------
        connect : {'fc', 'cf'}
            Connection matrix to update.

        segment : str
            Segment of context to update.

        item : int
            Absolute index of the item in the network.

        L : double
            Learning rate.
        """
        ind = self.c_ind[segment]
        if connect == 'fc':
            self.w_fc_exp[item, ind] += self.c[ind] * L
        elif connect == 'cf':
            self.w_cf_exp[item, ind] += self.c[ind] * L
        else:
            raise ValueError(f'Invalid connection: {connect}')

    def study(self, segment, item_list, B, Lfc, Lcf, distract_segment=None,
              distract_list=None, distract_B=None):
        """
        Study a list of items.

        Parameters
        ----------
        segment : str
            Segment representing the items to be presented.

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

        distract_segment : str, optional
            Segment representing distraction trials.

        distract_list : numpy.array, optional
            Distraction item indices relative to the segment.

        distract_B : float or numpy.array
            Context updating rate for each distraction event before
            and after each study event. If an array, must be of length
            n_items + 1. Distraction will not be presented on trials i
            where distract_B[i] is zero.
        """
        ind = self.f_ind[segment].start + item_list
        ind = ind.astype(np.dtype('i'))
        if not isinstance(B, np.ndarray):
            B = np.tile(B, item_list.shape).astype(float)
        if not isinstance(Lfc, np.ndarray):
            Lfc = np.tile(Lfc, item_list.shape).astype(float)
        if not isinstance(Lcf, np.ndarray):
            Lcf = np.tile(Lcf, item_list.shape).astype(float)

        if distract_segment is None or distract_list is None:
            distract_ind = np.ndarray(shape=(0,), dtype=np.dtype('i'))
        else:
            distract_ind = self.f_ind[distract_segment].start + distract_list
            distract_ind = distract_ind.astype(np.dtype('i'))

        if distract_B is None:
            distract_B = np.zeros(item_list.shape[0] + 1, dtype=float)
        elif not isinstance(distract_B, np.ndarray):
            distract_B = np.tile(distract_B,
                                 item_list.shape[0] + 1).astype(float)

        operations.study(self.w_fc_exp, self.w_fc_pre,
                         self.w_cf_exp, self.c, self.c_in,
                         self.f, ind, B, Lfc, Lcf, distract_ind, distract_B)

    def _p_recall_cython(self, segment, recalls, B, T, p_stop, amin=0.000001):
        rec_ind = self.f_ind[segment]
        n_item = rec_ind.stop - rec_ind.start
        exclude = np.zeros(n_item, dtype=np.dtype('i'))
        p = np.zeros(len(recalls) + 1)
        recalls = np.array(recalls, dtype=np.dtype('i'))
        support = np.zeros(n_item)
        operations.p_recall(rec_ind.start, n_item, recalls,
                            self.w_fc_exp, self.w_fc_pre,
                            self.w_cf_exp, self.w_cf_pre,
                            self.f, self.c, self.c_in,
                            exclude, amin, B, T, p_stop, support, p)
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
        w_cf = self.w_cf_exp[rec_ind, :] + self.w_cf_pre[rec_ind, :]

        recalls = []
        exclude = np.zeros(w_cf.shape[0], dtype=bool)
        n_item = self.n_f_segment[segment]
        item_ind = np.arange(n_item)
        for i in range(n_item):
            # stop recall with some probability
            if np.random.rand() < p_stop[i]:
                break

            # project the current state of context; assume nonzero support
            support = np.dot(w_cf, self.c)
            support[support < amin] = amin

            # scale based on choice parameter, set recalled items to zero
            strength = np.exp((2 * support) / T)
            strength[exclude] = 0

            # select item for recall proportionate to support
            p_recall = strength / np.sum(strength)
            recall = np.random.choice(item_ind, p=p_recall)
            recalls.append(recall)
            exclude[recall] = 1

            # integrate context associated with the item into context
            self.integrate(segment, recall, B)
        return recalls

    def plot(self):
        """Plot the current state of the network."""
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 5, figure=fig)
        ax = {'c': fig.add_subplot(gs[0, 1:4]),
              'f': fig.add_subplot(gs[2, 1:4]),
              'w_fc_pre': fig.add_subplot(gs[1, 0]),
              'w_fc_exp': fig.add_subplot(gs[1, 1]),
              'w_cf_exp': fig.add_subplot(gs[1, 3]),
              'w_cf_pre': fig.add_subplot(gs[1, 4])}
        for name, h in ax.items():
            mat = getattr(self, name)
            if mat.ndim == 1:
                mat = mat[None, :]
            h.matshow(mat)
            h.set_axis_off()
            h.set_title(name)
