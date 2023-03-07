## Introduction

This repository is intended for Cython+OpenMP practice of matrix-level pairwise distance calculation. It has been empirically proved that Cython version is faster than Numba version, and is close to Scipy implementation.

We also try to integrate C++20 new features `jthread` in our implementation for acceleration.



## Environment

Windows/Linux



## Quick Start

*Update: this demo has been upload on [Pypi](https://pypi.org/project/pairwise-distance-euclidean/) and can be installed via pip.*

To use new features, you need to install `gcc-10` and `g++-10`. Please following these commands for installation.

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10
sudo apt install g++-10
```

After that, navigate to the folder of `fastdist_cython` and compile the source code with 

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

* With `jthread` in 16-core GCP Virtual Machine

  | Scipy (cdist) | Ours (Cython+jthread*16) |
  | ------------- | ------------------------ |
  | 0.1582 sec    | 0.0014 sec               |

* Input for pairwise distance calculation: matrix: (512, 1024)

  | [Fastdist](https://github.com/talboger/fastdist) (Numba) | Scipy (pdist) | Ours (Cython) |
  | -------------------------------------------------------- | ------------- | ------------- |
  | 0.3154 sec                                               | 0.1266 sec    | 0.1236 sec    |

  