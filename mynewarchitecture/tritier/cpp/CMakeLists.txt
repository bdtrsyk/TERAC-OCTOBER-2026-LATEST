cmake_minimum_required(VERSION 3.20)
project(tritier_sycl LANGUAGES CXX)

# Invoke through an Intel oneAPI environment, e.g.
# cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icx
find_package(IntelSYCL REQUIRED)
add_library(tritier_sycl STATIC src/tritier_gpu_sycl.cpp)
target_include_directories(tritier_sycl PUBLIC include)
target_compile_features(tritier_sycl PUBLIC cxx_std_20)
target_link_libraries(tritier_sycl PUBLIC IntelSYCL::SYCL_CXX)
