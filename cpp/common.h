#pragma once

#include <complex>
#include <vector>
#include <random>
#include <ctime>

std::mt19937_64 rnd(239);

#define TIME (clock() * 1.0 / CLOCKS_PER_SEC)

typedef long double Float;
typedef std::vector<int> State;
typedef std::complex<Float> Complex;
typedef std::vector<std::vector<Complex>> Matrix;
typedef std::vector<std::vector<Float>> FloatMatrix;

std::istream& operator>>(std::istream& in, Complex& c){
    Float a, b, sgn = 1;
    char symb;
    in >> a;
    in >> symb;
    if (symb == '-')
        sgn = -1;
    in >> b;
    in >>symb;
    c = Complex(a, sgn*b);
    return in;
}

std::istream& operator>>(std::istream& in, Matrix& M){
    int n;
    in >> n;
    M = Matrix(n);
    char symb;
    for (int i = 0; i < n; ++i){
        in >> symb;
        M[i] = std::vector<Complex>(M.size());
        for (int j = 0; j < n; ++j){
            in >> M[i][j];
            in >>symb;
        }
        if (i != M.size() - 1)
            in >> symb;
    }
    return in;
}

std::ostream& operator<<(std::ostream& out, const State& state) {
    out << "(";
    for (size_t i = 0; i < state.size(); ++i) {
        out << state[i];
        if (i + 1 != state.size()) {
            out << ",";
        }
    }
    out << ")";
    return out;
}