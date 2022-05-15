#pragma once

#include "common.h"

#include <iostream>
#include <algorithm>

std::vector<std::vector<Float>> RandomNormal(int n) {
    std::normal_distribution<Float> normal(0, 1);
    std::vector<std::vector<Float>> A(n);
    for (int i = 0; i < n; ++i) {
        A[i].reserve(n);
        for (int j = 0; j < n; ++j) {
            A[i].emplace_back(normal(rnd));
        }
    }
    return A;
}

Complex operator*(const std::vector<Complex>& left, const std::vector<Complex>& right) {
    Complex result = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        result += left[i] * std::conj(right[i]);
    }
    return result;
}

std::vector<Complex> operator+(const std::vector<Complex>& left, const std::vector<Complex>& right) {
    std::vector<Complex> result(left);
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] += right[i];
    }
    return result;
}

std::vector<Complex> operator-(const std::vector<Complex>& left, const std::vector<Complex>& right) {
    std::vector<Complex> result(left);
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] -= right[i];
    }
    return result;
}

std::vector<Complex> operator*(const std::vector<Complex>& left, Complex coef) {
    std::vector<Complex> result(left);
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] *= coef;
    }
    return result;
}

std::vector<Complex> operator/(const std::vector<Complex>& left, Complex coef) {
    std::vector<Complex> result(left);
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] /= coef;
    }
    return result;
}

Float SquaredNorm(const std::vector<Complex>& v) {
    Float res = 0;
    for (auto x : v) {
        res += std::norm(x);
    }
    return res;
}

Matrix RandomUnitary(int n) {
    auto A_real = RandomNormal(n);
    auto A_img = RandomNormal(n);
    Matrix A(n);
    for (int i = 0; i < n; ++i) {
        A[i].reserve(n);
        for (int j = 0; j < n; ++j) {
            A[i].emplace_back(Complex(A_real[i][j], A_img[i][j]));
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i] = A[i] - (A[j] * (A[i] * A[j]));
        }
        A[i] = A[i] / sqrt(SquaredNorm(A[i]));
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            std::swap(A[i][j], A[j][i]);
        }
    }
    return A;
}

Matrix Conjugate(const Matrix& A) {
    int n = A.size();
    int m = A[0].size();
    Matrix result(m, std::vector<Complex>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[j][i] = std::conj(A[i][j]);
        }
    }
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    assert(A[0].size() == B.size());
    int n = A.size();
    int m = A[0].size();
    int k = B[0].size();
    Matrix result(n, std::vector<Complex>(k, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int h = 0; h < k; ++h) {
                result[i][h] += A[i][j] * B[j][h];
            }
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& out, const Matrix& A) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            out << A[i][j] << " ";
        }
        if (i + 1 != A.size()) {
            out << "\n";
        }
    }
    return out;
}

int SampleFromDensity(const std::vector<Float>& p) {
    std::uniform_real_distribution<Float> uniform(0, 1);
    Float x = uniform(rnd);
    Float sum = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        sum += p[i];
        if (x <= sum) {
            return i;
        }
    }
    assert(false);
}

int SampleFromDensityFast(const std::vector<Float>& cum_p) {
    std::uniform_real_distribution<Float> uniform(0, 1);
    Float x = uniform(rnd);
    return std::lower_bound(cum_p.begin(), cum_p.end(), x) - cum_p.begin();
}
