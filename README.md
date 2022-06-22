## Introduction

This repository is intended for Cython+OpenMP practice of matrix-level pairwise distance calculation. It has been empirically proved that Cython version is faster than Numba version, and is close to Scipy implementation.



## Environment

Windows



## Quick Start

First, navigate to the folder of `fastdist_cython` and compile the source code with 

```python
python setup.py build_ext --inplace
```

Once you get the success message, run the test code with

```python
python test.py
```



## Performance

* Inputs for matrix to matrix distance calculation: matrix_a: (200, 1024) & matrix_b: (1500, 1024)

  | [Fastdist](https://github.com/talboger/fastdist) (Numba) | Scipy (cdist)                       | Ours (Cython)      |
  |-------------------------------------------------------|-------------------------------------| ------------------------------------- |
  | 0.4691 sec                                                  | 0.3250 sec                          | 0.3225 sec    |

  

* Input for pairwise distance calculation: matrix: (512, 1024)

  | [Fastdist](https://github.com/talboger/fastdist) (Numba) | Scipy (pdist) | Ours (Cython) |
  | -------------------------------------------------------- | ------------- | ------------- |
  | 0.3154 sec                                               | 0.1266 sec    | 0.1236 sec    |

  