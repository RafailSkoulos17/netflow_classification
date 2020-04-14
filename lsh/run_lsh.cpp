//
// Created by rafail on 1/24/20.
//

/**
 * @file rhplsh_test.cpp
 *
 * @brief Example of using Random Hyperplane LSH index for L2 distance.
 */

#include <rh_lsh.h>
#include <metric.h>
#include <iostream>
#include <tuple>
#include <fstream>
#include <string>

int write_example() {
    std::ofstream myfile("example.txt");
    if (myfile.is_open()) {
        myfile << "This is a line.\n";
        myfile << "This is another line.\n";
        myfile.close();
    } else std::cout << "Unable to open file";
    return 0;
}

void gen_random(char *s, const int len) {
    static const char alphanum[] =
            "abcd";
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
}

std::vector<std::vector<std::string>> create_data(int dim1, int dim2, int len) {
    std::vector<std::vector<std::string>> data;
    for (int i = 0; i != dim1; ++i) {
        std::vector<std::string> d1 = {};
        for (int j = 0; j != dim2; ++j) {
            char s[10];
            gen_random(s, len);
//            std::cout << "D: " << s << std::endl;
            std::string str(s);
            d1.emplace_back(s);
        }
        data.push_back(d1);
    }
    return data;
}

int main(int argc, char const *argv[]) {

    srand(42);
    char file[] = "out.txt";
    std::cout << "Example of using Random Hyperplane LSH" << std::endl << std::endl;
    typedef std::string DATATYPE;
    lshbox::timer timer;
    timer.restart();

    lshbox::rhpLsh<DATATYPE> mylsh;
    std::cout << "Created LSH ..." << std::endl;
    int len = 4;
    std::vector<std::vector<std::string>> data = create_data(100, 20, len);

    std::cout << "Created data ..." << std::endl;

    lshbox::rhpLsh<DATATYPE>::Parameter param;
    param.M = 521; // Hash table size
    param.L = 10; // Number of hash tables
    param.D = len;
//    param.D = 4; // Dimension of the vector
    param.N = 100; // Binary code bytes
    mylsh.reset(param);
    for (const auto &v : data) {
        mylsh.hash(v);
    }
    mylsh.save(file);
    write_example();
    std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl;
}