#pragma once

#include "common.h"
#include "sampler.h"
#include "lib.h"
#include "random_utils.h"

#include <tuple>

Float RatioTest(Sampler& sampler, const std::vector<State>& states) {
    Float log_sum = 1;
    for (const auto& state : states) {
        log_sum += log(sampler.GetPsProb(state, false));
        log_sum -= log(sampler.GetPdProb(state, false));
    }
    return (1 / (1 + exp(-log_sum)));
}

Float Fidelity(Sampler& sampler, const std::vector<State>& states) {
    StatesMap mp;
    for (const auto& state : states) {
        mp[state]++;
    }
    Float result = 0;
    for (const auto& val : mp) {
        result += sqrt((sampler.GetPsProb(val.first, false) * val.second) / states.size());
    }
    return result;
}

std::tuple<Float, Float, Float> AverageScoreValueMis(int n, int t1, int t2, int k, int num_tests = 100) {
    Float sum_score = 0;
    Float sum_sqr_score = 0;
    Float time_to_sample = 0;
    for (int _ = 0; _ < num_tests; ++_) {
        Sampler sampler(n * n, n);
        Float start_time = TIME;
        auto states = sampler.SampleMis(k, t1, t2);
        time_to_sample += (TIME - start_time);
        Float score = Fidelity(sampler, states);
        sum_score += score;
        sum_sqr_score += score * score;
    }
    sum_score /= num_tests;
    sum_sqr_score /= num_tests;
    time_to_sample /= num_tests;
    return std::make_tuple(sum_score, sqrt(sum_sqr_score - sum_score * sum_score), time_to_sample);
}

std::tuple<Float, Float, Float> AverageScoreValueStandard(int n, int k, int num_tests = 100) {
    Float sum_score = 0;
    Float sum_sqr_score = 0;
    Float time_to_sample = 0;
    for (int _ = 0; _ < num_tests; ++_) {
        Sampler sampler(n * n, n);
        Float start_time = TIME;

        auto all_states = GenOutputStates(n * n, n);
        auto p = sampler.GetDensity(all_states);
        for (size_t i = 1; i < p.size(); ++i) {
            p[i] += p[i - 1];
        }
        std::vector<State> states;
        states.reserve(k);
        for (int i = 0; i < k; ++i) {
            states.push_back(all_states[SampleFromDensityFast(p)]);
        }
        time_to_sample += (TIME - start_time);

        Float score = Fidelity(sampler, states);
        sum_score += score;
        sum_sqr_score += score * score;
    }
    sum_score /= num_tests;
    sum_sqr_score /= num_tests;
    time_to_sample /= num_tests;
    return std::make_tuple(sum_score, sqrt(sum_sqr_score - sum_score * sum_score), time_to_sample);
}
