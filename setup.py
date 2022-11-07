from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

os.environ["CC"] = "gcc-10"
os.environ["CXX"] = "g++-10"

compile_args = ['-std=c++17', '-fopenmp', "-O3", "-pthread"]
linker_flags = ['-fopenmp', "-pthread"]

module = Extension('pairwise_distance_euclidean',
                   ['pairwise_distance_euclidean.pyx'],
                   language='c++',
                   include_dirs=[np.get_include()],
                   extra_compile_args=compile_args,
                   extra_link_args=linker_flags,)
                   #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])

setup(
    name='pairwise_distance_euclidean',
    ext_modules=cythonize(module),
    gdb_debug=False
)
