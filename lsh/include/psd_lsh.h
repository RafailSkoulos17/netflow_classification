
/**
 * @file psdlsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
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
 #include <lsh/include/basis.h>
#include <list>


namespace lshbox {
/**
 * Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
 */
    template<typename DATATYPE = int>
    class psdLsh {
    public:
        struct Parameter {
            /// Hash table size
            unsigned M;
            /// Number of hash tables
            unsigned L;
            /// Dimension of the vector, it can be obtained from the instance of Matrix
            unsigned D;
            /// Index mode, you can choose 1(CAUCHY) or 2(GAUSSIAN)
            unsigned T;
            /// Window size
            float W;
        };

        psdLsh() {}

        psdLsh(const Parameter &param_) {
            reset(param_);
        }

        ~psdLsh() {}

        /**
         * Reset the parameter setting
         *
         * @param param_ A instance of psdLsh<DATATYPE>::Parametor, which contains
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
        void insert(unsigned key,  std::vector <DATATYPE> domin);

        /**
         * get the hash value of a vector.
         *
         * @param k     The idx of the table
         * @param domin The pointer to the vector
         * @return      The hash value
         */
        unsigned getHashVal(unsigned k,  std::vector <DATATYPE> domin);


        std::vector <std::map<unsigned, std::vector < std::string>> >
        tables;


    private:
        Parameter param;
        std::vector<float> rndBs;
        std::vector <std::vector<float>> stableArray;

    };
}

// ------------------------- implementation -------------------------
template<typename DATATYPE>
void lshbox::psdLsh<DATATYPE>::reset(const Parameter &param_) {
    srand(1993);
    param = param_;
    tables.resize(param.L);
    stableArray.resize(param.L);
    std::mt19937 rng(unsigned(1993));
    std::uniform_real_distribution<float> ur(0, param.W);
    switch (param.T) {
        case CAUCHY: {
            std::cauchy_distribution<float> cd;
            for (std::vector < std::vector < float > > ::iterator iter = stableArray.begin(); iter != stableArray.end();
            ++iter)
            {
                for (unsigned i = 0; i != param.D; ++i) {
                    iter->push_back(cd(rng));
                }
                rndBs.push_back(ur(rng));
            }
            return;
        }
        case GAUSSIAN: {
            std::normal_distribution<float> nd;
            for (std::vector < std::vector < float > > ::iterator iter = stableArray.begin(); iter != stableArray.end();
            ++iter)
            {
                for (unsigned i = 0; i != param.D; ++i) {
                    iter->push_back(nd(rng));
                }
                rndBs.push_back(ur(rng));
            }
            return;
        }
        default: {
            return;
        }
    }
}

template<typename DATATYPE>
void lshbox::psdLsh<DATATYPE>::hash(std::vector <DATATYPE> data) {
    insert(0, data);

}

template<typename DATATYPE>
void lshbox::psdLsh<DATATYPE>::insert(unsigned key,  std::vector <DATATYPE> domin) {
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
unsigned lshbox::psdLsh<DATATYPE>::getHashVal(unsigned k, std::vector <DATATYPE> domin) {
    float sum(0);
    int iters;
    if (domin.size() < param.D) iters=domin.size();
    else iters = param.D;
    for (unsigned i = 0; i != iters; ++i) {
        sum += domin[i] * stableArray[k][i];
    }
    unsigned hashVal = unsigned(std::floor((sum + rndBs[k]) / param.W)) % param.M;
    return hashVal;
}
