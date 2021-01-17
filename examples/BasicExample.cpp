#include "blaze/Blaze.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

template <typename T>
using tensor2d = blaze::DynamicMatrix<float>;

constexpr int N = 1024 / 2;

int main() {
  {
    /* Initialization test! */
    tensor2d<float> A = {{1, 2}, {3, 4}}, B = {{1, 2}, {3, 4}};
    tensor2d<float> C = A * blaze::trans(B);
    auto X            = A * B;

    A.resize(N, N);
    B.resize(N, N);
    C.resize(N, N);

    std::iota(A.data(), A.data() + size(A), 1);
    std::iota(B.data(), B.data() + size(B), 1);
    std::fill(C.data(), C.data() + size(C), 0);

    // naive matrix-matrix multiplication
    auto start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++) {
          C(i, j) += A(i, k) * B(k, j);
        }
    auto end = std::chrono::system_clock::now();
    std::cout << "naive: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f
              << std::endl;
    std::cout << "size of C is " << size(C) << std::endl;
    std::cout << "sum: " << reduce(C, blaze::Add()) << std::endl;

    // blaze matrix-matrix multiplication
    std::fill(C.data(), C.data() + size(C), 0);
    start = std::chrono::system_clock::now();
    C     = A * B;
    end   = std::chrono::system_clock::now();

    std::cout << "blaze: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f
              << std::endl;
    std::cout << "size of C is " << size(C) << std::endl;
    std::cout << "sum: " << reduce(C, blaze::Add()) << std::endl;
  }

  {
    /* Matrix-Vector Multiplication */
    blaze::DynamicMatrix<float> M                      = {{1, 2}, {3, 4}};
    blaze::DynamicVector<float, blaze::columnVector> V = {5, 6};

    // result is Vector(not a matrix)
    blaze::DynamicVector<float> Z = blaze::eval(M * V);

    for (auto e : Z)
      std::cout << e << std::endl;
  }
}
