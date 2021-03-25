# cython: language_level=3, profile=False, boundscheck=False, wraparound=False, embedsignature=True

cimport cython
from libc.math cimport sqrt, exp


@cython.profile(False)
cdef inline double calc_rho(double cdot, double B):
    """
    Calculate context integration scaling factor.
    
    Parameters
    ----------
    cdot
        Dot product between :math:`c` and :math:`c^{IN}`.
    
    B
        Beta parameter weighting :math:`c^{IN}`.
    
    Returns
    -------
    rho
        Scaling factor for :math:`c`.   
    """
    rho = sqrt(1 + (B * B) * ((cdot * cdot) - 1)) - (B * cdot)
    return rho


cpdef integrate_context(double [:] c,
                        double[:] c_in,
                        double B,
                        int [:] c_ind):
    """
    Integrate context input.
    
    Parameters
    ----------
    c
        Context state :math:`c`.
    
    c_in
        Input to context :math:`c^{IN}`
    
    B
        :math:`\beta` parameter weighting :math:`c`.
    
    c_ind
        Start and end indices of context to update.
    """
    cdef double cdot = 0
    cdef int i
    for i in range(c_ind[0], c_ind[1]):
        cdot += c[i] * c_in[i]
    rho = calc_rho(cdot, B)

    for i in range(c_ind[0], c_ind[1]):
        c[i] = rho * c[i] + B * c_in[i]


cpdef integrate(double [:, :] w_fc_exp,
                const double [:, :] w_fc_pre,
                double [:] c,
                double [:] c_in,
                double [:] f,
                int item,
                int [:, :] c_ind,
                double [:] B):
    """
    Integrate context input associated with an item into context.
    
    Parameters
    ----------
    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.
    
    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.
    
    c
        Context state :math:`\mathbf{c}`.
    
    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.
    
    f
        Item representation :math:`\mathbf{f}`.
    
    item
        Index of item to present.
    
    c_ind
        Start and end indices of the context sublayer.
    
    B
        :math:`\\beta` parameter.
    """
    cdef Py_ssize_t n_f = f.shape[0]
    cdef Py_ssize_t n_s = c_ind.shape[0]
    cdef int i
    cdef int j

    # set item unit
    for i in range(n_f):
        if i == item:
            f[i] = 1
        else:
            f[i] = 0

    # set c_in
    cdef double sum_squares
    cdef double norm
    for i in range(n_s):
        sum_squares = 0
        for j in range(c_ind[i, 0], c_ind[i, 1]):
            c_in[j] = w_fc_exp[item, j] + w_fc_pre[item, j]
            sum_squares += c_in[j] * c_in[j]

        # normalize the vector to have an L2 norm of 1
        norm = sqrt(sum_squares)
        for j in range(c_ind[i, 0], c_ind[i, 1]):
            c_in[j] /= norm

        # integrate
        integrate_context(c, c_in, B[i], c_ind[i])


cpdef present(double [:, :] w_fc_exp,
              const double [:, :] w_fc_pre,
              double [:, :] w_cf_exp,
              double [:] c,
              double [:] c_in,
              double [:] f,
              int item,
              int [:, :] c_ind,
              double [:] B,
              double [:] Lfc,
              double [:] Lcf):
    """
    Present an item and associate with context.
    
    Parameters
    ----------
    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.
    
    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.
    
    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.
    
    c
        Context state :math:`\mathbf{c}`.
    
    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.
    
    f
        Item representation :math:`\mathbf{f}`.
    
    item
        Index of item to present.
    
    c_ind
        Start and end indices of the context sublayer.
    
    B
        :math:`\\beta` parameter.
    
    Lfc
        :math:`L^{FC}` parameter.
    
    Lcf
        :math:`L^{CF}` parameter.
    """
    cdef Py_ssize_t n_f = f.shape[0]
    cdef Py_ssize_t n_s = c_ind.shape[0]
    cdef int i
    cdef int j

    # retrieve item context and integrate into current context
    integrate(w_fc_exp, w_fc_pre, c, c_in, f, item, c_ind, B)

    # learn the association between f and c
    for i in range(n_s):
        if Lfc[i] > 0:
            for j in range(c_ind[i, 0], c_ind[i, 1]):
                w_fc_exp[item, j] += Lfc[i] * c[j]

    for i in range(n_s):
        if Lcf[i] > 0:
            for j in range(c_ind[i, 0], c_ind[i, 1]):
                w_cf_exp[item, j] += Lcf[i] * c[j]


cpdef study(
    double [:, :] w_fc_exp,
    const double [:, :] w_fc_pre,
    double [:, :] w_cf_exp,
    double [:] c,
    double [:] c_in,
    double [:] f,
    const int [:] item_list,
    int [:, :] c_ind,
    double [:, :] B,
    double [:, :] Lfc,
    double [:, :] Lcf,
):
    """
    Simulate study of a list.
    
    Parameters
    ----------
    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.
    
    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.
    
    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.
    
    c
        Context state :math:`\mathbf{c}`.
    
    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.
    
    f
        Item representation :math:`\mathbf{f}`.
    
    item_list
        Indices of items to present.
    
    c_ind
        Start and end indices of the context sublayer.
    
    B
        :math:`\\beta` parameter.
    
    Lfc
        :math:`L^{FC}` parameter.
    
    Lcf
        :math:`L^{CF}` parameter.
    """
    cdef Py_ssize_t n = item_list.shape[0]
    for i in range(n):
        present(w_fc_exp, w_fc_pre, w_cf_exp, c, c_in, f,
                item_list[i], c_ind, B[i], Lfc[i], Lcf[i])


cpdef study_distract(
    double [:, :] w_fc_exp,
    const double [:, :] w_fc_pre,
    double [:, :] w_cf_exp,
    double [:] c,
    double [:] c_in,
    double [:] f,
    const int [:] item_list,
    int [:, :] c_ind,
    double [:, :] B,
    double [:, :] Lfc,
    double [:, :] Lcf,
    const int [:] distract_list,
    double [:, :] distract_B
):
    """
    Simulate study of a list with distraction.
    
    Parameters
    ----------
    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.
    
    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.
    
    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.
    
    c
        Context state :math:`\mathbf{c}`.
    
    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.
    
    f
        Item representation :math:`\mathbf{f}`.
    
    item_list
        Indices of items to present.
    
    c_ind
        Start and end indices of the context sublayer.
    
    B
        :math:`\\beta` parameter.
    
    Lfc
        :math:`L^{FC}` parameter.
    
    Lcf
        :math:`L^{CF}` parameter.
    
    distract_list
        Indices of distraction items to present.
    
    distract_B
        :math:`\\beta_\mathrm{distract}` parameter.
    """
    cdef Py_ssize_t n = item_list.shape[0]
    for i in range(n):
        integrate(w_fc_exp, w_fc_pre, c, c_in, f,
                  distract_list[i], c_ind, distract_B[i])
        present(w_fc_exp, w_fc_pre, w_cf_exp, c, c_in, f,
                item_list[i], c_ind, B[i], Lfc[i], Lcf[i])
    integrate(w_fc_exp, w_fc_pre, c, c_in, f,
              distract_list[n], c_ind, distract_B[n])


cpdef cue_item(int n,
               int n_f,
               const double [:, :] w_cf_pre,
               double [:, :] w_cf_exp,
               const double [:, :] w_ff_pre,
               double [:, :] w_ff_exp,
               double [:] f_in,
               double [:] c,
               int [:] exclude,
               int [:] recalls,
               int output):
    """
    Cue an item based on context.
    
    Parameters
    ----------
    n
        Start index of the item segment being recalled from.
    
    n_f
        Number of units in the item segment.
    
    w_cf_pre
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{pre}`.
    
    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.
    
    w_ff_pre
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{pre}`.
    
    w_ff_exp
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{exp}`.
        
    f_in
        Item input :math:`\mathbf{f}^\mathrm{IN}`.
    
    c
        Context representation :math:`\mathbf{c}`.
    
    exclude
        Vector of item indices to exclude from recall.
    
    recalls
        Item indices of recalls in output order.
    
    output
        Output position, starting from zero.
    """
    cdef Py_ssize_t n_c = w_cf_exp.shape[1]
    cdef int i
    cdef int j

    for i in range(n_f):
        f_in[n + i] = 0
        if exclude[i]:
            continue

        # support from context cuing
        for j in range(n_c):
            f_in[n + i] += ((w_cf_exp[n + i, j] + w_cf_pre[n + i, j]) * c[j])

        if output > 0:
            # support from the previously recalled item
            f_in[n + i] += (w_ff_exp[n + recalls[output - 1], n + i] +
                            w_ff_pre[n + recalls[output - 1], n + i])


cpdef apply_softmax(int n,
                    int n_f,
                    double [:] f_in,
                    int [:] exclude,
                    double amin,
                    double T):
    """
    Apply softmax rule to item activation.
    
    Parameters
    ----------
    n
        Start index of the item segment being recalled from.
    
    n_f
        Number of units in the item segment.
    
    f_in
        Item input :math:`\mathbf{f}^\mathrm{IN}`.
    
    exclude
        Vector of item indices to exclude from recall.
    
    amin
        Minimum item activation for non-excluded items.
    
    T
        Temperature parameter of the softmax function.
    """
    cdef int i
    for i in range(n_f):
        if exclude[i]:
            continue

        # ensure minimal support for each item
        if f_in[n + i] < amin:
            f_in[n + i] = amin

        # apply softmax
        f_in[n + i] = exp((2 * f_in[n + i]) / T)


def item_match(
    int n,
    int n_f,
    const double [:, :] w_fc_pre,
    double [:, :] w_fc_exp,
    double [:] c,
    double [:] match,
):
    """Calculate match between context and potential item recalls."""
    cdef Py_ssize_t n_c = w_fc_exp.shape[1]
    cdef int i
    cdef int j

    for i in range(n_f):
        match[n + i] = 0

        # support from context cuing
        for j in range(n_c):
            match[n + i] += ((w_fc_exp[n + i, j] + w_fc_pre[n + i, j]) * c[j])


def apply_expit(
    int n,
    int n_f,
    double [:] match,
    double A1,
    double A2,
):
    """Apply expit function to calculate acceptance probability."""
    cdef int i
    for i in range(n_f):
        match[n + i] = 1 / (1 + exp(-(A1 + A2 * match)))


def p_recall(int start,
             int n_f,
             int [:] recalls,
             double [:, :] w_fc_exp,
             const double [:, :] w_fc_pre,
             double [:, :] w_cf_exp,
             const double [:, :] w_cf_pre,
             double [:, :] w_ff_exp,
             const double [:, :] w_ff_pre,
             double [:] f,
             double [:] f_in,
             double [:] c,
             double [:] c_in,
             int [:, :] c_ind,
             int [:] exclude,
             double amin,
             double [:, :] B,
             double T,
             const double [:] p_stop,
             double [:] p):
    """
    Calculate the likelihood of each recall in a sequence.

    Parameters
    ----------
    start
        Start unit for the item segment to recall from.

    n_f
        Number of units in the item segment.

    recalls
        Item indices of recalls in output order.

    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.

    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.

    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.

    w_cf_pre
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{pre}`.

    w_ff_exp
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{exp}`.

    w_ff_pre
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{pre}`.

    f
        Item representation :math:`\mathbf{f}`.

    f_in
        Item activation input :math:`\mathbf{f}^\mathrm{IN}`.

    c
        Context state :math:`\mathbf{c}`.

    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.

    c_ind
        Start and end indices of the context sublayer.

    exclude
        Vector of item indices to exclude from recall.

    amin
        Minimum item activation for non-excluded items.

    B
        :math:`\\beta` parameter.

    T
        Temperature parameter of the softmax function.

    p_stop
        Probability of stopping by output position.

    p
        Likelihood of each recall and the stopping event.
    """
    cdef Py_ssize_t n_r = recalls.shape[0]
    cdef int i
    cdef int j
    cdef double total

    for i in range(n_r):
        # calculate support for each item
        cue_item(start, n_f, w_cf_pre, w_cf_exp, w_ff_pre, w_ff_exp,
                 f_in, c, exclude, recalls, i)
        apply_softmax(start, n_f, f_in, exclude, amin, T)

        total = 0
        for j in range(n_f):
            total += f_in[start + j]

        # calculate probability of this recall
        p[i] = (f_in[start + recalls[i]] / total) * (1 - p_stop[i])

        # update context
        if i < (n_r - 1):
            exclude[recalls[i]] = 1
            integrate(w_fc_exp, w_fc_pre, c, c_in, f, recalls[i], c_ind, B[i])
    p[n_r] = p_stop[n_r]


def p_recall_match(
    int start,
    int n_f,
    int [:] recalls,
    double [:, :] w_fc_exp,
    const double [:, :] w_fc_pre,
    double [:, :] w_cf_exp,
    const double [:, :] w_cf_pre,
    double [:, :] w_ff_exp,
    const double [:, :] w_ff_pre,
    double [:] f,
    double [:] f_in,
    double [:] c,
    double [:] c_in,
    int [:, :] c_ind,
    int [:] exclude,
    double amin,
    double [:, :] B,
    double T,
    const double [:] p_stop,
    double [:] p,
    double [:] match,
    double A1,
    double A2,
):
    """
    Calculate the likelihood of each recall in a sequence.

    Parameters
    ----------
    start
        Start unit for the item segment to recall from.

    n_f
        Number of units in the item segment.

    recalls
        Item indices of recalls in output order.

    w_fc_exp
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{exp}`.

    w_fc_pre
        Weight matrix :math:`\mathbf{M}^{FC}_\mathrm{pre}`.

    w_cf_exp
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{exp}`.

    w_cf_pre
        Weight matrix :math:`\mathbf{M}^{CF}_\mathrm{pre}`.

    w_ff_exp
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{exp}`.

    w_ff_pre
        Weight matrix :math:`\mathbf{M}^{FF}_\mathrm{pre}`.

    f
        Item representation :math:`\mathbf{f}`.

    f_in
        Item activation input :math:`\mathbf{f}^\mathrm{IN}`.

    c
        Context state :math:`\mathbf{c}`.

    c_in
        Context input :math:`\mathbf{c}^\mathrm{IN}`.

    c_ind
        Start and end indices of the context sublayer.

    exclude
        Vector of item indices to exclude from recall.

    amin
        Minimum item activation for non-excluded items.

    B
        :math:`\\beta` parameter.

    T
        Temperature parameter of the softmax function.

    p_stop
        Probability of stopping by output position.

    p
        Likelihood of each recall and the stopping event.

    match
        Degree of match between current context and context associated
        with an item.

    A1
        Intercept mapping match to an expit function.

    A2
        Slope mapping match to an expit function.
    """
    cdef Py_ssize_t n_r = recalls.shape[0]
    cdef int i
    cdef int j
    cdef double total
    cdef double p_recall

    for i in range(n_r):
        # calculate support for each item
        cue_item(start, n_f, w_cf_pre, w_cf_exp, w_ff_pre, w_ff_exp,
                 f_in, c, exclude, recalls, i)
        apply_softmax(start, n_f, f_in, exclude, amin, T)

        total = 0
        for j in range(n_f):
            total += f_in[start + j]

        # calculate probability of this recall
        item_match(start + recalls[i], 1, w_fc_pre, w_fc_exp, match)
        apply_expit(start + recalls[i], 1, match, A1, A2)
        p_recall = (f_in[start + recalls[i]] / total) * match[start + recalls[i]]
        p[i] = p_recall * (1 - p_stop[i])

        # update context
        if i < (n_r - 1):
            exclude[recalls[i]] = 1
            integrate(w_fc_exp, w_fc_pre, c, c_in, f, recalls[i], c_ind, B[i])
    p[n_r] = p_stop[n_r]
