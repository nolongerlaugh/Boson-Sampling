#include "common.h"
#include "random_utils.h"
#include "lib.h"
#include "sampler.h"
#include "test.h"
#include "compare_test.h"

#include <fstream>


int main() {

    /*auto A = RandomUnitary(26);
    std::cout << A << std::endl;
    std::cout << std::endl;
    std::cout << (A * Conjugate(A)) << std::endl;
    std::cout << std::endl;
    std::cout << Permanent(A) << std::endl;*/
    for (int m=6; m < 8; ++m)
    {
        int n = m * m;
        std::cout << m << std::endl;
        Sampler sampler(n, m, true);
        //auto states = sampler.SampleMis(30000, 100, 50);
        //std::cout << RatioTest(sampler, states) << std::endl;
        auto base_result = AverageScoreValueStandard(m, 30000, 3);
        std::cout << m << ", " << "," << get<0>(base_result) << "+-" << get<1>(base_result) << ", time: " << get<2>(base_result) << std::endl;
        //std::cout << base_result << std::endl;
    }
    /*int n;
    std::cin >> n;
    Matrix M(n);
    std::cin >> M;
    //a = Complex(1, 2);
    std::cout << M << std::endl;*/
    /*std::ofstream mis_out("tests/mis_result.csv", std::ios::app);
    test_avg_score(mis_out);
    test_time_mis(mis_out);
    test_three_density();
    std::cout << "working time: " << TIME << std::endl;*/
    return 0;
}
