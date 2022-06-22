import pairwise_distance_euclidean
import numpy as np
import time
from scipy.spatial.distance import cdist, pdist, squareform
from fastdist import fastdist


if __name__ == "__main__":
    # test matrix to matrix distance
    matrix_a = np.random.randn(200, 1024)
    matrix_b = np.random.randn(1500, 1024)
    start = time.time()
    for _ in range(10):
        dist_cython = pairwise_distance_euclidean.euclidean_matrix_2_matrix(matrix_a, matrix_b)
    end = time.time()
    print("Cython running time is {:.4f} seconds.".format((end-start) / 10))
    start = time.time()
    for _ in range(10):
        dist_scipy = cdist(matrix_a, matrix_b)
    end = time.time()
    print("Scipy running time is {:.4f} seconds.".format((end-start) / 10))
    start = time.time()
    for _ in range(10):
        dist_fast = fastdist.matrix_to_matrix_distance(matrix_a, matrix_b, fastdist.euclidean, "euclidean")
    end = time.time()
    print("Fastdist running time is {:.4f} seconds.".format((end-start) / 10))
    assert np.allclose(dist_cython, dist_scipy)
    # test pairwise distance
    matrix_square = np.random.randn(512, 1024)
    start = time.time()
    for _ in range(10):
        pairwise_cython = pairwise_distance_euclidean.euclidean_pairwise(matrix_square)
    end = time.time()
    print("Cython running time is {:.4f} seconds.".format((end-start) / 10))
    start = time.time()
    for _ in range(10):
        pairwise_scipy = squareform(pdist(matrix_square))
    end = time.time()
    print("Scipy running time is {:.4f} seconds.".format((end-start) / 10))
    start = time.time()
    for _ in range(10):
        pairwise_fastdist = fastdist.matrix_pairwise_distance(matrix_square, fastdist.euclidean, "euclidean", True)
    end = time.time()
    print("Fastdist running time is {:.4f} seconds.".format((end - start) / 10))

