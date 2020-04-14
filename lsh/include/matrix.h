//
// Created by rafail on 1/24/20.
//

#ifndef MY_LSH_MATRIX_H
#define MY_LSH_MATRIX_H

#endif //MY_LSH_MATRIX_H
/**
 * @file matrix.h
 *
 * @brief Dataset management class.
 */
#pragma once
#include <fstream>
#include <vector>
#include <assert.h>
#include <string.h>
namespace lshbox
{
/**
 * Dataset management class. A dataset is maintained as a matrix in memory.
 *
 * The file contains N D-dimensional vectors of single precision floating point numbers.
 *
 * Such binary files can be accessed using lshbox::Matrix<double>.
 */
    template <class T>
    class Matrix
    {
        int dim;
        int N;
        T *dims;
    public:
        /**
         * Reset the size.
         *
         * @param _dim Dimension of each vector
         * @param _N   Number of vectors
         */
        void reset(int _dim, int _N)
        {
            dim = _dim;
            N = _N;
            if (dims != NULL)
            {
                delete [] dims;
            }
            dims = new T[dim * N];
        }
        Matrix(): dim(0), N(0), dims(NULL) {}
        Matrix(int _dim, int _N): dims(NULL)
        {
            reset(_dim, _N);
        }
        ~Matrix()
        {
            if (dims != NULL)
            {
                delete [] dims;
            }
        }
        /**
         * Access the ith vector.
         */
        const T *operator [] (int i) const
        {
            return dims + i * dim;
        }
        /**
         * Access the ith vector.
         */
        T *operator [] (int i)
        {
            return dims + i * dim;
        }
        /**
         * Get the dimension.
         */
        int getDim() const
        {
            return dim;
        }
        /**
         * Get the size.
         */
        int getSize() const
        {
            return N;
        }
        /**
         * Get the data.
         */
        T * getData() const
        {
            return dims;
        }
        /**
         * Load the Matrix from a binary file.
         */
        void load(const std::string &path)
        {
            std::ifstream is(path.c_str(), std::ios::binary);
            unsigned header[3];
            assert(sizeof header == 3 * 4);
            is.read((char *)header, sizeof(header));
            reset(header[2], header[1]);
            is.read((char *)dims, sizeof(T) * dim * N);
            is.close();
        }
        /**
         * Load the Matrix from std::vector<T>.
         *
         * @param vec  The reference of std::vector<T>.
         * @param _N   Number of vectors
         * @param _dim Dimension of each vector
         */
        void load(std::vector<T> &vec, int _N, int _dim)
        {
            reset(_dim, _N);
            memcpy(dims, (void*)&vec[0], sizeof(T) * dim * N);
        }
        /**
         * Load the Matrix from T*.
         *
         * @param source The pointer to T*.
         * @param _N     Number of vectors
         * @param _dim   Dimension of each vector
         */
        void load(T *source, int _N, int _dim)
        {
            reset(_dim, _N);
            memcpy(dims, source, sizeof(T) * dim * N);
        }
        /**
         * Save the Matrix as a binary file.
         */
        void save(const std::string &path)
        {
            std::ofstream os(path.c_str(), std::ios::binary);
            unsigned header[3];
            header[0] = sizeof(T);
            header[1] = N;
            header[2] = dim;
            os.write((char *)header, sizeof header);
            os.write((char *)dims, sizeof(T) * dim * N);
            os.close();
        }
        Matrix(const std::string &path): dims(NULL)
        {
            load(path);
        }
        Matrix(const Matrix& M): dims(NULL)
        {
            reset(M.getDim(), M.getSize());
            memcpy(dims, M.getData(), sizeof(T) * dim * N);
        }
        Matrix& operator = (const Matrix& M)
        {
            dims = NULL;
            reset(M.getDim(), M.getSize());
            memcpy(dims, M.getData(), sizeof(T) * dim * N);
            return *this;
        }
        /**
         * An accessor class to be used with LSH index.
         */
        class Accessor
        {
            const Matrix &matrix_;
            std::vector<bool> flags_;
        public:
            typedef unsigned Key;
            typedef const T *Value;
            typedef T DATATYPE;
            Accessor(const Matrix &matrix): matrix_(matrix)
            {
                flags_.resize(matrix_.getSize());
            }
            void reset()
            {
                flags_.clear();
                flags_.resize(matrix_.getSize());
            }
            bool mark(unsigned key)
            {
                if (flags_[key])
                {
                    return false;
                }
                flags_[key] = true;
                return true;
            }
            const T *operator () (unsigned key)
            {
                return matrix_[key];
            }
        };
    };
}