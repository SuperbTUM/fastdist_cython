#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include "omp.h"


template<typename T, bool col_major=false>
class MatrixView{
public:
    T* data_pointer;
    const long nrow;
    const long ncol;

    MatrixView(T *data_pointer, const long nrow, const long ncol) : data_pointer(data_pointer), nrow(nrow),
                                                                    ncol(ncol) {}

    T &operator()(const int row, const int col) {
        if (col_major) {
            return data_pointer[row + col * nrow];
        } else {
            return data_pointer[col + row * ncol];
        }
    }

    T operator()(const int row, const int col) const {
        if (col_major) {
            return data_pointer[row + col * nrow];
        } else {
            return data_pointer[col + row * ncol];
        }
    }
};


void get_matrix_to_matrix_dist(double *a, double *b, double *res, long a_rows, long b_rows, long vec_dim){
    const MatrixView<double, false> a_view(a, a_rows, vec_dim);
    const MatrixView<double, false> b_view(b, b_rows, vec_dim);
    int idx = 0;
    #pragma omp taskloop for maxcpus(4)
    #pragma omp taskloop for schedtype(static)
    #pragma omp parallel for num_threads(4)
    for(auto i = 0; i < a_rows; i++){
        for(auto j = 0; j < b_rows; j++){
            double cur_sum = 0.0;
            #pragma omp taskloop for reduction(cur_sum)
            for(auto k = 0; k < vec_dim; k++){
                double diff = a_view(i, k) - b_view(j, k);
                cur_sum += diff * diff;
            }
            *(res + idx++) = sqrt(cur_sum);
        }
    }
}

void get_pairwise_dist(double *a, long a_rows, long vec_dim, double *res){
    const MatrixView<double, false> a_view(a, a_rows, vec_dim);
    auto i = 0;
    #pragma omp taskloop for maxcpus(4)
    #pragma omp taskloop for schedtype(static)
    #pragma omp parallel for num_threads(4)
    for(; i < a_rows; i++){
        for(auto j = 0; j < i; j++){
            double cur_sum = 0.0;
            #pragma omp taskloop for reduction(cur_sum)
            for(auto k = 0; k < vec_dim; k++){
                double diff = (a_view(i, k) - a_view(j, k));
                cur_sum += diff * diff;
            }
            res[a_rows * i + j] = sqrt(cur_sum);
            res[a_rows * j + i] = sqrt(cur_sum);
        }
    }
}