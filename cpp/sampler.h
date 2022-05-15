#pragma once

#include "common.h"
#include "random_utils.h"
#include "lib.h"

#include <unordered_map>

struct StateHash {
    size_t operator()(const State& state) const {
        size_t result = 0;
        for (auto x : state) {
            result = result * 239 + (x + 1);
        }
        return result;
    }
};

typedef std::unordered_map<State, Float, StateHash> StatesMap;

class Sampler {
public:
    Sampler(int m, int n, bool need_matrix = false) : m(m), n(n), U(RandomUnitary(m)), A(m, std::vector<Float>(n, 0)), fact(n + 1) {
        if (need_matrix)
        {
            std::cin >> U;
        }
        for (int i = 0; i < m; ++i) {
            U[i].resize(n);
            for (int j = 0; j < n; ++j) {
                A[i][j] = std::norm(U[i][j]);
            }
        }
        fact[0] = 1;
        for (int i = 1; i <= n; ++i) {
            fact[i] = fact[i - 1] * i;
        }
    }

    std::vector<State> SampleMis(int k, int t1 = 1, int t2 = 1) {
        State x_t = SampleFromPd();
        std::vector<State> result;
        result.reserve(k);
        for (int i = 0; i < t1; ++i) {
            x_t = GenOnePointMis(x_t);
        }
        int l = 0;
        while (result.size() < k) {
            l++;
            x_t = GenOnePointMis(x_t);
            if (l == t2) {
                result.push_back(x_t);
                l = 0;
            }
        }
        return result;
    }

    std::vector<Float> GetDensity(const std::vector<State>& states, bool save=false) {
        std::vector<Float> result(states.size());
        for (size_t i = 0; i < states.size(); ++i) {
            result[i] = GetPsProb(states[i], save);
        }
        return result;
    }

    // distinguishable
    Float GetPdProb(const State& state, bool save = true) {
        auto it = pds.find(state);
        if (it != pds.end()) {
            return it->second;
        }
        Float result = Permanent(AsMatrix(A, state));
        for (int x : state) {
            result /= fact[x];
        }
        if (save) {
            pds[state] = result;
        }
        return result;
    }

    // indistinguishable
    Float GetPsProb(const State& state, bool save = true) {
        auto it = ps.find(state);
        if (it != ps.end()) {
            return it->second;
        }
        Float result = std::norm(Permanent(AsMatrix(U, state)));
        for (int x : state) {
            result /= fact[x];
        }
        if (save) {
            ps[state] = result;
        }
        return result;
    }

    State SampleFromPd() {
        State result(m, 0);
        std::vector<Float> row(m, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                row[j] = A[j][i];
            }
            result[SampleFromDensity(row)]++;
        }
        return result;
    }

    const Matrix& GetU(){
        return U;
    }

private:
    State GenOnePointMis(const State& x_t) {
        State x_cur = SampleFromPd();
        Float a1 = GetPsProb(x_cur) / GetPsProb(x_t);
        Float a2 = GetPdProb(x_t) / GetPdProb(x_cur);
        Float a = a1 * a2;
        if (a >= 1) {
            return x_cur;
        }
        std::uniform_real_distribution<Float> uniform(0, 1);
        if (uniform(rnd) > a) {
            return x_t;
        }
        return x_cur;
    }

private:
    int m, n;
    Matrix U;
    FloatMatrix A;
    std::vector<Float> fact;
    StatesMap ps, pds;
};
