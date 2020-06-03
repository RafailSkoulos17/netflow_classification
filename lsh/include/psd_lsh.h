
/**
 * @file psdlsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
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
 * Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
 *
 *
 * For more information on p-stable distribution based LSH, see the following reference.
 *
 *     Mayur Datar , Nicole Immorlica , Piotr Indyk , Vahab S. Mirrokni,
 *     Locality-sensitive hashing scheme based on p-stable distributions,
 *     Proceedings of the twentieth annual symposium on Computational geometry, June
 *     08-11, 2004, Brooklyn, New York, USA.
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
         * Query the approximate nearest neighborholds.
         *
         * @param domin   The pointer to the vector
         * @param scanner Top-K scanner, use for scan the approximate nearest neighborholds
         */
        template<typename SCANNER>
        void query(const DATATYPE *domin, SCANNER &scanner);

        /**
         * get the hash value of a vector.
         *
         * @param k     The idx of the table
         * @param domin The pointer to the vector
         * @return      The hash value
         */
        unsigned getHashVal(unsigned k,  std::vector <DATATYPE> domin);

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
        std::vector<float> rndBs;
        std::vector <std::vector<float>> stableArray;

    };
}

// ------------------------- implementation -------------------------
template<typename DATATYPE>
void lshbox::psdLsh<DATATYPE>::reset(const Parameter &param_) {
    param = param_;
    tables.resize(param.L);
    stableArray.resize(param.L);
    std::mt19937 rng(unsigned(std::time(0)));
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
//    progress_display pd(data.getSize());
//    for (unsigned i = 0; i != data.getSize(); ++i)
//    {
//        insert(i, data[i]);
//        ++pd;
//    }
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

//template<typename DATATYPE>
//template<typename SCANNER>
//void lshbox::psdLsh<DATATYPE>::query(const DATATYPE *domin, SCANNER &scanner) {
//    scanner.reset(domin);
//    for (unsigned k = 0; k != param.L; ++k) {
//        unsigned hashVal = getHashVal(k, domin);
//        if (tables[k].find(hashVal) != tables[k].end()) {
//            for (std::vector<unsigned>::iterator iter = tables[k][hashVal].begin();
//                 iter != tables[k][hashVal].end(); ++iter) {
//                scanner(*iter);
//            }
//        }
//    }
//    scanner.topk().genTopk();
//}

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

template<typename DATATYPE>
void lshbox::psdLsh<DATATYPE>::load(const std::string &file) {
    std::ifstream in(file, std::ios::binary);
    in.read((char *) &param.M, sizeof(unsigned));
    in.read((char *) &param.L, sizeof(unsigned));
    in.read((char *) &param.D, sizeof(unsigned));
    in.read((char *) &param.W, sizeof(float));
    tables.resize(param.L);
    stableArray.resize(param.L);
    rndBs.resize(param.L);
    in.read((char *) &rndBs[0], sizeof(float) * param.L);
    for (unsigned i = 0; i != param.L; ++i) {
        stableArray[i].resize(param.D);
        in.read((char *) &stableArray[i][0], sizeof(float) * param.D);
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
void lshbox::psdLsh<DATATYPE>::save(const std::string &file) {
    std::ofstream out(file, std::ios::binary);
    out.write((char *) &param.M, sizeof(unsigned));
    out.write((char *) &param.L, sizeof(unsigned));
    out.write((char *) &param.D, sizeof(unsigned));
    out.write((char *) &param.W, sizeof(float));
    out.write((char *) &rndBs[0], sizeof(float) * param.L);
    for (int i = 0; i != param.L; ++i) {
        out.write((char *) &stableArray[i][0], sizeof(float) * param.D);
        unsigned count = unsigned(tables[i].size());
        out.write((char *) &count, sizeof(unsigned));
        for (std::map < unsigned, std::vector < std::string > > ::iterator iter = tables[i].begin(); iter !=
                                                                                                  tables[i].end();
        ++iter)
        {
            unsigned target = iter->first;
            out.write((char *) &target, sizeof(unsigned));
            unsigned length = unsigned(iter->second.size());
            out.write((char *) &length, sizeof(unsigned));
            out.write((char *) &((iter->second)[0]), sizeof(unsigned) * length);
        }
    }
    out.close();
}