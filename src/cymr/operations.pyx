# cython: profile=True

cimport cython
from libc.math cimport sqrt


@cython.profile(False)
cdef inline double calc_rho(double cdot, double B):
    rho = sqrt(1 + pow(B, 2) * (pow(cdot, 2) - 1)) - (B * cdot)
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
