cmake_minimum_required(VERSION 2.8.12)
project(kalman)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
#find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(kalman src/kalman.cpp)
target_link_libraries(kalman PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX)

#target_compile_definitions(kalman PUBLIC EIGEN_USE_MKL_ALL)
set(CMAKE_CXX_FLAGS "-ffast-math ${CMAKE_CXX_FLAGS}")
