//
// Created by rafail on 1/24/20.
//

/**
 * @file rhplsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on Random Hyperplanes.
 *
 * Inspired by https://github.com/RSIA-LIESMARS-WHU/LSHBOX
 */
#pragma once

#include <ctime>
#include <map>
#include <vector>
#include <random>
#include <iostream>
#include <functional>
#include <list>


namespace lshbox {
/**
 * Locality-Sensitive Hashing Scheme Based on Random Hyperplane.
 */
    template<typename DATATYPE = int>
    class rhpLsh {
    public:
        struct Parameter {
            /// Hash table size
            unsigned M;
            /// Number of hash tables
            unsigned L;
            /// Dimension of the vector, it can be obtained from the instance of Matrix
            unsigned D;
            /// Binary code bytes
            unsigned N;
        };

        rhpLsh() {}

        rhpLsh(const Parameter &param_) {
            reset(param_);
        }

        ~rhpLsh() {}

        /**
         * Reset the parameter setting
         *
         * @param param_ A instance of rhpLsh<DATATYPE>::Parametor, which contains
         * the necessary parameters
         */
        void reset(const Parameter &param_);

        /**
         * Hash the dataset.
         *
         * @param data A instance of Matrix<DATATYPE>, it is the search dataset.
         */
        void hash(std::vector <DATATYPE> data);

        /**
         * Insert a vector to the index.
         *
         * @param key   The sequence number of vector
         * @param domin The pointer to the vector
         */
        void insert(unsigned key, std::vector <DATATYPE> domin);


        /**
         * get the hash value of a vector.
         *
         * @param k     The idx of the table
         * @param domin The pointer to the vector
         * @return      The hash value
         */
        unsigned getHashVal(unsigned k, std::vector <DATATYPE> domin);




        std::vector <std::map<unsigned, std::vector < std::string>> >
        tables;


    private:
        Parameter param;
        std::vector <std::vector<unsigned>> rndArray;
        std::vector <std::vector<std::vector < float>> >
        uosArray;
    };
}

// ------------------------- implementation -------------------------
template<typename DATATYPE>
void lshbox::rhpLsh<DATATYPE>::reset(const Parameter &param_) {
//    unsigned int my_seed = 1993;
    srand(1993);
    param = param_;
    tables.resize(param.L);
    uosArray.resize(param.L);
    rndArray.resize(param.L);
//    std::mt19937 rng(unsigned(std::time(0)));
    std::mt19937 rng(unsigned(1993));
    std::normal_distribution<float> nd;
    std::uniform_int_distribution<unsigned> usArray(0, param.M - 1);
    for (auto ithRb = uosArray.begin();
         ithRb != uosArray.end(); ++ithRb) {
        ithRb->resize(param.N);
        for (auto iter = ithRb->begin(); iter != ithRb->end(); ++iter) {
            for (unsigned k = 0; k != param.D; ++k) {
                iter->push_back(nd(rng));
            }
        }
    }
    for (auto iter = rndArray.begin(); iter != rndArray.end(); ++iter) {
        for (unsigned i = 0; i != param.N; ++i) {
            iter->push_back(usArray(rng));
        }
    }
}


template<typename DATATYPE>
void lshbox::rhpLsh<DATATYPE>::hash(std::vector <DATATYPE> data) {
    insert(0, data);
}

template<typename DATATYPE>
void lshbox::rhpLsh<DATATYPE>::insert(unsigned key, std::vector <DATATYPE> domin) {
    for (unsigned k = 0; k != param.L; ++k) {
        unsigned hashVal = getHashVal(k, domin);
        std::string name = "";
        for (auto it = domin.begin(); it != domin.end(); ++it) {
            int curr = *it;
            name = name + "_" + std::to_string(curr);
        }
        tables[k][hashVal].push_back(name);
    }
}


template<typename DATATYPE>
unsigned lshbox::rhpLsh<DATATYPE>::getHashVal(unsigned k, std::vector <DATATYPE> domin) {
    unsigned sum(0);
    for (unsigned i = 0; i != param.N; ++i) {
        float flag(0);
        int iters;
        if (domin.size() < param.D) iters=domin.size();
        else iters = param.D;
        for (unsigned j = 0; j != iters; ++j) {
            flag += uosArray[k][i][j] * domin[j];
        }
        if (flag > 0) {
            sum += rndArray[k][i];
        }
    }
    unsigned hashVal = sum % param.M;

    return hashVal;
}