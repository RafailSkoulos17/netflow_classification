cmake_minimum_required(VERSION 2.8.4)
project(flexfringe)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0")
set(flexfringe_SOURCE_DIR /home/rafail/Documents/MS_Computer_Science/Thesis/netflow_classification)

add_custom_target(flexfringe COMMAND make -C ${flexfringe_SOURCE_DIR}
        CLION_EXE_DIR=${PROJECT_BINARY_DIR})