# distutils: language=c++
cimport cython
import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "matrix_2_matrix.h":
    void get_matrix_to_matrix_dist(double*, double*, double*, const long, const long, const long)
    void get_pairwise_dist(double*, const long, const long, double*)

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def euclidean_matrix_2_matrix(np.ndarray[double, ndim=2, mode="c"] ca, np.ndarray[double, ndim=2, mode="c"] cb):
    # cdef np.ndarray[double, ndim=2, mode="c"] ca = np.ascontiguousarray(a, dtype=np.float64)
    # cdef np.ndarray[double, ndim=2, mode="c"] cb = np.ascontiguousarray(b, dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] res = np.ascontiguousarray(
        np.zeros(shape=(ca.shape[0], cb.shape[0]), dtype=np.float64))
    # res = np.random.randn(a.shape[0], b.shape[0])
    get_matrix_to_matrix_dist(&ca[0,0], &cb[0,0], &res[0,0], ca.shape[0], cb.shape[0], ca.shape[1])
    return res

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef euclidean_pairwise(a):
    cdef np.ndarray[double, ndim=2, mode="c"] ca = np.ascontiguousarray(a, dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] res = np.ascontiguousarray(
        np.zeros(shape=(a.shape[0], a.shape[0]), dtype=np.float64))
    get_pairwise_dist(&ca[0,0], a.shape[0], a.shape[1], &res[0,0])
    return res
