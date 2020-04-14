cmake_minimum_required(VERSION 3.3)
project (lsh)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --verbose -std=c++11 -Wall -O3 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")


set(BUILD lsh)
set(SOURCE_FILES run_lsh_2.cpp)

set(INCLUDE_DIR ./include)
include_directories (${INCLUDE_DIR})

add_executable(${BUILD} ${SOURCE_FILES})




