#pragma once
#include <fstream>

void test_avg_score(std::ofstream &mis_out)
{
    //std::ofstream mis_out("tests/mis_result.csv", std::ios::app);
    mis_out << "n,t1,t2,k,average_score,score_std,average_time" << std::endl;

    std::ofstream base_out("tests/base_result.csv", std::ios::app);
    base_out << "n,k,average_score,score_std,average_time" << std::endl;

    for (int n = 2; n <= 13; n++) {
        std::cerr << n << " started!" << std::endl;
        int k = 100;

        int num_tests = 100;
        if (n >= 6) {
            num_tests = 10;
        }
        if (n < 8) {
            auto base_result = AverageScoreValueStandard(n, k, num_tests);
            base_out << n << "," << k << "," << get<0>(base_result) << "," << get<1>(base_result) << "," << get<2>(base_result) << std::endl;
        }
        for (int t1 : {10, 50, 100, 200}) {
            for (int t2 : {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}) {
                std::cerr << n << " " << t1 << " " << t2 << " started!" << std::endl;

                int num_tests2 = 100;
                if (n >= 6) {
                    num_tests2 = 10;
                }

                auto mis_res = AverageScoreValueMis(n, t1, t2, k, num_tests2);
                mis_out << n << "," << t1 << "," << t2 << "," << k << "," << get<0>(mis_res) << "," << get<1>(mis_res) << "," << get<2>(mis_res) << std::endl;

                std::cerr << n << " " << t1 << " " << t2 << " done!" << std::endl;
            }
        }
    }
}

void test_time_mis(std::ofstream &mis_out)
{
    for (int n = 14; n <= 22; ++n)
    {
        std::cerr << n << " started!" << std::endl;
        int t1 = 100;
        int t2 = 50;
        int k = 100;
        int num_tests2 = 1;
        auto mis_res = AverageScoreValueMis(n, t1, t2, k, num_tests2);
        mis_out << n << "," << t1 << "," << t2 << "," << k << "," << get<0>(mis_res) << "," << get<1>(mis_res) << "," << get<2>(mis_res) << std::endl;
        std::cerr << n << " done!" << std::endl;
    }
}

void test_three_density(){
    int n = 6;
    int k = 20000;
    Sampler sampler(n * n, n);
    std::vector<State> d_mis = sampler.SampleMis(20000, 100, 50);
    std::cout << "mis done!" << std::endl;

    std::vector<State> all_states = GenOutputStates(n * n, n);
    std::vector<Float> p = sampler.GetDensity(all_states);
    for (size_t i = 1; i < p.size(); ++i) {
        p[i] += p[i - 1];
    }
    std::vector<State> d_standart;
    d_standart.reserve(k);
    for (int i = 0; i < k; ++i) {
        d_standart.push_back(all_states[SampleFromDensityFast(p)]);
    }
    std::cout << "standart done!" << std::endl;

    std::vector<State> d_disting;
    d_disting.reserve(k);
    for (int i = 0; i < k; ++i) {
        d_disting.push_back(sampler.SampleFromPd());
    }

    std::cout << "distig done!" << std::endl;

    Float p_mis, p_standart, p_disting;
    std::ofstream mis_out("three_compare.csv");
    mis_out << "p_mis,p_standart,p_disting" << std::endl;
    for (int i = 0; i < k; ++i)
    {
        p_mis = -log(std::norm(Permanent(AsMatrix(sampler.GetU(), d_mis[i]))));
        p_standart = -log(std::norm(Permanent(AsMatrix(sampler.GetU(), d_standart[i]))));
        p_disting = -log(std::norm(Permanent(AsMatrix(sampler.GetU(), d_disting[i]))));
        mis_out << p_mis << "," << p_standart << "," << p_disting << std::endl;
    }
}

void ratio_test()
{
    std::ofstream mis_out("tests/ratio_test.csv");
    mis_out << "n,k,p_ind" << std::endl;
    for (int n = 2; n < 20; n++)
    {
        Sampler smp(n*n, n);
        for (int k = 1; k < 200; k += 10){
            std::vector<State> d_mis = smp.SampleMis(k, 100, 50);
            mis_out << n << "," << k << "," << RatioTest(smp, d_mis) << std::endl;
        }
        std::cout << n << " done!" << std::endl;
    }
}
