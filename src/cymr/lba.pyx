# cython: language_level=3, cdivision=True, profile=False

from libc.math cimport sqrt, exp, erf, pi


cdef double normpdf(double x):
    return (1 / sqrt(2 * pi)) * exp(-(x ** 2) / 2)


cdef double normcdf(double x):
    return (1 / 2.) * (1 + erf(x / sqrt(2)))


cdef double tpdf(double t, double A, double b, double v, double sv):
    """Probability distribution function over time."""
    cdef double g
    cdef double h
    cdef double f
    g = (b - A - t * v) / (t * sv)
    h = (b - t * v) / (t * sv)
    f = (-v * normcdf(g) + sv * normpdf(g) +
         v * normcdf(h) - sv * normpdf(h)) / A
    return f


cdef double tcdf(double t, double A, double b, double v, double s):
    """Cumulative distribution function over time."""
    cdef double e1
    cdef double e2
    cdef double e3
    cdef double e4
    cdef double F
    e1 = ((b - A - t * v) / A) * normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F


cpdef response_pdf(double t, int ind, double A, double b,
                  double [:] v, double s):
    """Probability density function of a response."""
    cdef Py_ssize_t n_v = v.shape[0]
    cdef double p_neg = 1
    cdef double p_n = 1
    cdef double pdf = 0
    cdef int i
    cdef int n

    if t <= 0:
        return pdf

    # probability of no accumulators finishing
    for i in range(n_v):
        p_neg *= normcdf(-v[i] / s)

    # probability of other accumulators not having finished
    for i in range(n_v):
        if i != ind:
            p_n *= 1 - tcdf(t, A, b, v[i], s)

    # probability of accumulator ind finishing at time t
    pdf = (tpdf(t, A, b, v[ind], s) * p_n) / (1 - p_neg)
    return pdf
