/**
 * @file rbslsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on Random Bits Sampling.
 */

#pragma once

#include <ctime>
#include <map>
#include <vector>
#include <random>
#include <iostream>
#include <functional>
//#include <matrix.h>
//#include <basis.h>
#include <list>
#include <string.h>
#include <algorithm>

namespace lshbox {
/**
 * Locality-Sensitive Hashing Scheme Based on Random Bits Sampling.
 *
 *
 * For more information on random bits sampling based LSH, see the following reference.
 *
 *     P. Indyk and R. Motwani. Approximate Nearest Neighbor - Towards Removing
 *     the Curse of Dimensionality. In Proceedings of the 30th Symposium on Theory
 *     of Computing, 1998, pp. 604-613.
 *
 *     A. Gionis, P. Indyk, and R. Motwani. Similarity search in high dimensions
 *     via hashing. Proceedings of the 25th International Conference on Very Large
 *     Data Bases (VLDB), 1999.
 */
    template<typename DATATYPE = int>
    class rbsLsh {
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
            /// The Difference between upper and lower bound of each dimension
            unsigned C;
        };

        rbsLsh() {}

        rbsLsh(const Parameter &param_) {
            reset(param_);
        }

        ~rbsLsh() {}

        /**
         * Reset the parameter setting
         *
         * @param param_ A instance of rbsLsh::Parametor, which contains the necessary
         * parameters
         */
        void reset(const Parameter &param_);

        /**
         * Hash the dataset.
         *
         * @param data A instance of Matrix<unsigned>, it is the search dataset.
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
        std::vector <std::vector<unsigned>> rndBits;
        std::vector <std::vector<unsigned>> rndArray;
    };
}

// ------------------------- implementation -------------------------
template<typename DATATYPE>
void lshbox::rbsLsh<DATATYPE>::reset(const Parameter &param_) {
    srand(1993);
    param = param_;
    tables.resize(param.L);
    rndBits.resize(param.L);
    rndArray.resize(param.L);
    std::mt19937 rng(unsigned(1993));
    std::uniform_int_distribution<unsigned> usBits(0, param.D * param.C - 1);
//    for (std::vector<std::vector<unsigned> >::iterator iter = rndBits.begin(); iter != rndBits.end(); ++iter) {
    for (auto iter = rndBits.begin(); iter != rndBits.end(); ++iter) {
        while (iter->size() != param.N) {
            unsigned target = usBits(rng);
            if (std::find(iter->begin(), iter->end(), target) == iter->end()) {
                iter->push_back(target);
            }
        }
        std::sort(iter->begin(), iter->end());
    }
    std::uniform_int_distribution<unsigned> usArray(0, param.M - 1);
//    for (std::vector<std::vector<unsigned> >::iterator iter = rndArray.begin(); iter != rndArray.end(); ++iter) {
    for (auto iter = rndArray.begin(); iter != rndArray.end(); ++iter) {
        for (unsigned i = 0; i != param.N; ++i) {
            iter->push_back(usArray(rng));
        }
    }
}

template<typename DATATYPE>
void lshbox::rbsLsh<DATATYPE>::hash(std::vector <DATATYPE> data) {
//    progress_display pd(data.size());
//    for (unsigned i = 0; i != data.size(); ++i) {
//        insert(i, data[i]);
//        ++pd;
//    }
    insert(0, data);

}

template<typename DATATYPE>
void lshbox::rbsLsh<DATATYPE>::insert(unsigned key, std::vector <DATATYPE> domin) {
//    for (unsigned k = 0; k != param.L; ++k) {
//        unsigned hashVal = getHashVal(k, domin);
//        tables[k][hashVal].push_back(domin);
//    }
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

//template<typename SCANNER>
//void lshbox::rbsLsh::query(const std::string *domin, SCANNER &scanner) {
//    scanner.reset(domin);
//    for (unsigned k = 0; k != param.L; ++k) {
//        unsigned hashVal = getHashVal(k, *domin);
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
unsigned lshbox::rbsLsh<DATATYPE>::getHashVal(unsigned k, std::vector <DATATYPE> domin) {
    unsigned sum(0), seq(0);
    for (auto it = rndBits[k].begin(); it != rndBits[k].end(); ++it) {
        if ((*it / param.C) < domin.size()) {
            if ((*it % param.C) <= unsigned(domin[*it / param.C])) {
                sum += rndArray[k][seq];
            }
        }
        ++seq;
    }

    unsigned hashVal = sum % param.M;
    return
            hashVal;
}

template<typename DATATYPE>
void lshbox::rbsLsh<DATATYPE>::load(const std::string &file) {
    std::ifstream in(file, std::ios::binary);
    in.read((char *) &param.M, sizeof(unsigned));
    in.read((char *) &param.L, sizeof(unsigned));
    in.read((char *) &param.D, sizeof(unsigned));
    in.read((char *) &param.C, sizeof(unsigned));
    in.read((char *) &param.N, sizeof(unsigned));
    tables.resize(param.L);
    rndBits.resize(param.L);
    rndArray.resize(param.L);
    for (unsigned i = 0; i != param.L; ++i) {
        rndBits[i].resize(param.N);
        rndArray[i].resize(param.N);
        in.read((char *) &rndBits[i][0], sizeof(unsigned) * param.N);
        in.read((char *) &rndArray[i][0], sizeof(unsigned) * param.N);
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
void lshbox::rbsLsh<DATATYPE>::save(const std::string &file) {
    std::ofstream out(file, std::ios::binary);
    out.write((char *) &param.M, sizeof(unsigned));
    out.write((char *) &param.L, sizeof(unsigned));
    out.write((char *) &param.D, sizeof(unsigned));
    out.write((char *) &param.C, sizeof(unsigned));
    out.write((char *) &param.N, sizeof(unsigned));
    for (int i = 0; i != param.L; ++i) {
        out.write((char *) &rndBits[i][0], sizeof(unsigned) * param.N);
        out.write((char *) &rndArray[i][0], sizeof(unsigned) * param.N);
        unsigned count = unsigned(tables[i].size());
        out.write((char *) &count, sizeof(unsigned));
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