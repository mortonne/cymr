# cython: language_level=3, profile=False

cimport cython
from libc.math cimport sqrt, exp


@cython.profile(False)
cdef inline double calc_rho(double cdot, double B):
    rho = sqrt(1 + (B * B) * ((cdot * cdot) - 1)) - (B * cdot)
    return rho


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef integrate_context(double [:] c,
                        double[:] c_in,
                        double B,
                        int [:] c_ind):
    cdef double cdot = 0
    cdef int i
    for i in range(c_ind[0], c_ind[1]):
        cdot += c[i] * c_in[i]
    rho = calc_rho(cdot, B)

    for i in range(c_ind[0], c_ind[1]):
        c[i] = rho * c[i] + B * c_in[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef integrate(double [:, :] w_fc_exp,
                const double [:, :] w_fc_pre,
                double [:] c,
                double [:] c_in,
                double [:] f,
                int item,
                int [:, :] c_ind,
                double [:] B):
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


@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef Py_ssize_t n = item_list.shape[0]
    for i in range(n):
        present(w_fc_exp, w_fc_pre, w_cf_exp, c, c_in, f,
                item_list[i], c_ind, B[i], Lfc[i], Lcf[i])


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef Py_ssize_t n = item_list.shape[0]
    for i in range(n):
        integrate(w_fc_exp, w_fc_pre, c, c_in, f,
                  distract_list[i], c_ind, distract_B[i])
        present(w_fc_exp, w_fc_pre, w_cf_exp, c, c_in, f,
                item_list[i], c_ind, B[i], Lfc[i], Lcf[i])
    integrate(w_fc_exp, w_fc_pre, c, c_in, f,
              distract_list[n], c_ind, distract_B[n])

@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef apply_softmax(int n,
                    int n_f,
                    double [:] f_in,
                    int [:] exclude,
                    double amin,
                    double T):
    cdef int i
    for i in range(n_f):
        if exclude[i]:
            continue

        # ensure minimal support for each item
        if f_in[n + i] < amin:
            f_in[n + i] = amin

        # apply softmax
        f_in[n + i] = exp((2 * f_in[n + i]) / T)


@cython.boundscheck(False)
@cython.wraparound(False)
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
             int [:] exclude,
             double amin,
             double [:] B,
             double T,
             const double [:] p_stop,
             double [:] p):
    cdef Py_ssize_t n_r = recalls.shape[0]
    cdef Py_ssize_t n_c = w_cf_exp.shape[1]
    cdef int i
    cdef int j
    cdef int k
    cdef double total = 0
    cdef double norm

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
        exclude[recalls[i]] = 1

        # update context
        if i < (n_r - 1):
            integrate(w_fc_exp, w_fc_pre, c, c_in, f, recalls[i], B[i])
    p[n_r] = p_stop[n_r]
