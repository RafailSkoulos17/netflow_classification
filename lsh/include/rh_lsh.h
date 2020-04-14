//
// Created by rafail on 1/24/20.
//

/**
 * @file rhplsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on Random Hyperplane.
 */
#pragma once

#include <ctime>
#include <map>
#include <vector>
#include <random>
#include <iostream>
#include <functional>
#include <lsh/include/matrix.h>
#include <lsh/include/basis.h>
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
         * Query the approximate nearest neighborholds.
         *
         * @param domin   The pointer to the vector
         * @param scanner Top-K scanner, use for scan the approximate nearest neighborholds
         */
        template<typename SCANNER>
        void query(const unsigned *domin, SCANNER &scanner);

        /**
         * get the hash value of a vector.
         *
         * @param k     The idx of the table
         * @param domin The pointer to the vector
         * @return      The hash value
         */
        unsigned getHashVal(unsigned k, std::vector <DATATYPE> domin);

        unsigned getHashValFromStr(unsigned k, const DATATYPE domin);


        /**
         * Load the index from binary file.
         *
         * @param file The path of binary file.
         */
        void load(const std::string &file);

        /**
         * Save the index as binary file.
         *
         * @param file The path of binary file.
         */
        void save(const std::string &file);

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
    srand(1717);
    param = param_;
    tables.resize(param.L);
    uosArray.resize(param.L);
    rndArray.resize(param.L);
//    std::mt19937 rng(unsigned(std::time(0)));
    std::mt19937 rng(unsigned(1717));
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
//    progress_display pd(data.size());
//    for (unsigned i = 0; i != data.size(); ++i) {
////        std::cout << "\nINSERTING  " << data[i] << "  WHICH IS THE " << i << " STRING OF THIS DATA" << std::endl;
//        insert(i, data[i]);
//        ++pd;
//    }

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
//
//template<typename DATATYPE>
//template<typename SCANNER>
//void lshbox::rhpLsh<DATATYPE>::query(const DATATYPE *domin, SCANNER &scanner) {
//    scanner.reset(domin);
//    for (unsigned k = 0; k != param.L; ++k) {
//        unsigned hashVal = getHashVal(k, domin);
//        if (tables[k].find(hashVal) != tables[k].end()) {
//            for (auto iter = tables[k][hashVal].begin();
//                 iter != tables[k][hashVal].end(); ++iter) {
//                scanner(*iter);
//            }
//        }
//    }
//    scanner.topk().genTopk();
//}


template<typename DATATYPE>
unsigned lshbox::rhpLsh<DATATYPE>::getHashVal(unsigned k, std::vector <DATATYPE> domin) {
//    std::cout << "GETTING ITS HASH VALUE" << std::endl;
    unsigned sum(0);
    for (unsigned i = 0; i != param.N; ++i) {
//        std::cout << "ITERATE FOR THE " << i << " BINARY VALUE" << std::endl;
        float flag(0);
        int iters;
        if (domin.size() < param.D) iters=domin.size();
        else iters = param.D;
        for (unsigned j = 0; j != iters; ++j) {
            flag += uosArray[k][i][j] * domin[j];
//            flag += uosArray[k][i][j] * domin;
        }
        if (flag > 0) {
            sum += rndArray[k][i];
        }
    }
    unsigned hashVal = sum % param.M;
//    std::cout << "HASHVAL: " << hashVal << std::endl;
//    std::cout << std::endl;


    return hashVal;
}

template<typename DATATYPE>
unsigned lshbox::rhpLsh<DATATYPE>::getHashValFromStr(unsigned k, const DATATYPE domin) {
//    std::cout << "GETTING ITS HASH VALUE" << std::endl;
    unsigned sum(0);
    for (unsigned i = 0; i != param.N; ++i) {
//        std::cout << "ITERATE FOR THE " << i << " BINARY VALUE" << std::endl;
        float flag(0);
        for (unsigned j = 0; j != param.D; ++j) {
            flag += uosArray[k][i][j] * (domin[j] & 0xff) * 100;
        }
        if (flag > 0) {
            sum += rndArray[k][i];
        }
    }
    unsigned hashVal = sum % param.M;
//    std::cout << "HASHVAL: " << hashVal << std::endl;
//    std::cout << std::endl;


    return hashVal;
}

template<typename DATATYPE>
void lshbox::rhpLsh<DATATYPE>::load(const std::string &file) {
    std::ifstream in(file, std::ios::binary);
    in.read((char *) &param.M, sizeof(unsigned));
    in.read((char *) &param.L, sizeof(unsigned));
    in.read((char *) &param.D, sizeof(unsigned));
    in.read((char *) &param.N, sizeof(unsigned));
    tables.resize(param.L);
    uosArray.resize(param.L);
    rndArray.resize(param.L);
    for (unsigned i = 0; i != param.L; ++i) {
        rndArray[i].resize(param.N);
        uosArray[i].resize(param.N);
        in.read((char *) &rndArray[i][0], sizeof(unsigned) * param.N);
        for (unsigned j = 0; j != param.N; ++j) {
            uosArray[i][j].resize(param.D);
            in.read((char *) &uosArray[i][j][0], sizeof(float) * param.D);
        }
        unsigned count;
        in.read((char *) &count, sizeof(unsigned));
        for (unsigned j = 0; j != count; ++j) {
            unsigned target;
            in.read((char *) &target, sizeof(unsigned));
            unsigned length;
            in.read((char *) &length, sizeof(unsigned));
            tables[i][target].resize(length);
            in.read((char *) &(tables[i][target][0]), sizeof(unsigned) * length);
        }
    }
    in.close();
}

template<typename DATATYPE>
void lshbox::rhpLsh<DATATYPE>::save(const std::string &file) {
    std::ofstream out(file, std::ios::binary);
    out.write((char *) &param.M, sizeof(unsigned));
    out.write((char *) &param.L, sizeof(unsigned));
    out.write((char *) &param.D, sizeof(unsigned));
    out.write((char *) &param.N, sizeof(unsigned));
    for (unsigned i = 0; i != param.L; ++i) {
        out.write((char *) &rndArray[i][0], sizeof(unsigned) * param.N);
        for (unsigned j = 0; j != param.N; ++j) {
            out.write((char *) &uosArray[i][j][0], sizeof(float) * param.D);
        }
        unsigned count = unsigned(tables[i].size());
        out.write((char *) &count, sizeof(unsigned));
//        for (std::map<unsigned, std::vector<unsigned> >::iterator iter = tables[i].begin();
        for (auto iter = tables[i].begin();
             iter != tables[i].end(); ++iter) {
            unsigned target = iter->first;
            out.write((char *) &target, sizeof(unsigned));
            unsigned length = unsigned(iter->second.size());
            out.write((char *) &length, sizeof(unsigned));
            out.write((char *) &((iter->second)[0]), sizeof(unsigned) * length);
        }
    }
    out.close();
}