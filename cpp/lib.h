#pragma once

#include "common.h"

#include <functional>

template <typename T>
T Permanent(const std::vector<std::vector<T>>& A) {
    int n = A.size();
    std::vector<T> dp(1 << n, 0);
    dp[0] = 1;
    for (int mask = 0; mask < (1 << n); ++mask) {
        int k = __builtin_popcount(mask);
        for (int i = 0; i < n; i++) {
            if ((mask >> i) & 1) {
                dp[mask] += dp[mask ^ (1 << i)] * A[k - 1][i];
            }
        }
    }
    return dp.back();
}

template <typename T>
std::vector<std::vector<T>> AsMatrix(const std::vector<std::vector<T>>& A, const State& s) {
    int n = A[0].size();
    std::vector<std::vector<T>> result(n, std::vector<T>(n, 0));
    int row_id = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        for (int j = 0; j < s[i]; ++j) {
            for (int x = 0; x < n; ++x) {
                result[row_id][x] = A[i][x];
            }
            ++row_id;
        }
    }
    return result;
}

std::vector<State> GenOutputStates(int m, int n) {
    std::vector<State> result;
    State cur(m);

    std::function<void(int,int)> gen;
    gen = [&](int i, int s) {
        if (i == m - 1) {
            cur[i] = s;
            result.push_back(cur);
            return;
        }
        for (int x = 0; x <= s; ++x) {
            cur[i] = x;
            gen(i + 1, s - x);
        }
    };

    gen(0, n);

    return result;
}
