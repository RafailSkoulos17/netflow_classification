//
// Created by rafail on 1/24/20.
//

/**
 * @file rhplsh_test.cpp
 *
 * @brief Example of using Random Hyperplane LSH index for L2 distance.
 */

#include <rbs_lsh.h>
#include <metric.h>
#include <iostream>
#include <tuple>
#include <fstream>
#include <string>


void generate_random_numbers(int r1, int r2, vector<int> &seq) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution <std::mt19937::result_type> dist(1, 18); // distribution in range [r1,r2]

    std::uniform_int_distribution <std::mt19937::result_type> length(1, 10);

    for (int i = 0; i != length(rng); ++i) {
        seq.push_back(dist(rng))
    }
}

int main(int argc, char const *argv[]) {

    srand(42);
    char file[] = "out.txt";
    int v1, v2, v3, v4;
    std::cout << "Example of using Random Bits Sampling LSH" << std::endl << std::endl;
    lshbox::timer timer;
    std::cout << "LOAD TIME: " << timer.elapsed() << "s." << std::endl;
    std::cout << "CONSTRUCTING INDEX ..." << std::endl;
    timer.restart();

//    std::string data = "abcd";

    lshbox::rbsLsh mylsh;
    std::cout << "Created LSH ..." << std::endl;
    lshbox::rbsLsh::Parameter param;
    param.M = 100; // Hash table size
    param.L = 1; // Number of hash tables
    param.D = 10; // Dimension of the vector
    param.C = 25; // The Difference between upper and lower bound of each dimension
    param.N = 100; // Binary code bytes
    mylsh.reset(param);

    for (int i = 0; i != 1000; ++i) {
        std::vector<int> seq;
        generate_random_numbers(1, 18, seq);
        mylsh.hash(seq);
    }
    std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl;
}