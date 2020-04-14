# cython: profile=True

cimport cython
from libc.math cimport sqrt, exp


@cython.profile(False)
cdef inline double calc_rho(double cdot, double B):
    rho = sqrt(1 + (B * B) * ((cdot * cdot) - 1)) - (B * cdot)
    return rho


@cython.boundscheck(False)
@cython.wraparound(False)
def update(double [:] c, double[:] c_in, double B):
    cdef Py_ssize_t n = c.shape[0]
    cdef double cdot = 0
    cdef int i
    for i in range(n):
        cdot += c[i] * c_in[i]
    rho = calc_rho(cdot, B)

    for i in range(n):
        c[i] = rho * c[i] + B * c_in[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef present(const double [:, :] w_fc_exp,
            const double [:, :] w_fc_pre,
            double [:] c,
            double [:] c_in,
            double [:] f,
            int item,
            double B):
    cdef Py_ssize_t n_f = f.shape[0]
    cdef Py_ssize_t n_c = c.shape[0]
    cdef int i

    # set item unit
    for i in range(n_f):
        if i == item:
            f[i] = 1
        else:
            f[i] = 0

    # set c_in
    for i in range(n_c):
        c_in[i] = w_fc_exp[item, i] + w_fc_pre[item, i]

    # get the vector length of the input to context
    cdef double sum_squares = 0
    for i in range(n_c):
        sum_squares += c_in[i] * c_in[i]
    cdef double norm = sqrt(sum_squares)
    for i in range(n_c):
        c_in[i] /= norm

    # calculate scaling factor
    cdef double cdot = 0
    for i in range(n_c):
        cdot += c[i] * c_in[i]
    rho = calc_rho(cdot, B)

    # integrate context
    for i in range(n_c):
        c[i] = rho * c[i] + B * c_in[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def p_recall(int start,
             int n_f,
             const int [:] recalls,
             const double [:, :] w_fc_exp,
             const double [:, :] w_fc_pre,
             const double [:, :] w_cf_exp,
             const double [:, :] w_cf_pre,
             double [:] f,
             double [:] c,
             double [:] c_in,
             int [:] exclude,
             double amin,
             double B,
             double T,
             const double [:] p_stop,
             double [:] support,
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
        for j in range(n_f):
            support[j] = 0
            if exclude[j]:
                continue
            for k in range(n_c):
                support[j] += ((w_cf_exp[start + j, k] + w_cf_pre[start + j, k])
                               * c[k])
                if support[j] < amin:
                    support[j] = amin
            support[j] = exp((2 * support[j]) / T)

        # sum of support for all items
        total = 0
        for j in range(n_f):
            total += support[j]

        # calculate probability of this recall
        p[i] = (support[recalls[i]] / total) * (1 - p_stop[i])
        exclude[recalls[i]] = 1

        # update context
        if i < (n_r - 1):
            present(w_fc_exp, w_fc_pre, c, c_in, f, recalls[i], B)
    p[n_r] = p_stop[n_r]
