
cmake_minimum_required (VERSION 3.9)
project ("PFAC-CUDA" LANGUAGES CXX CUDA)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")

add_executable(playground icalcu.cu)

add_executable(pfac pfac_main.cpp
					pfac.h
					pfac.cu
					pfac_match.cu
					pfac_table.h
					pfac_table.cu
					pfac_texture.cuh
					pfac_kernel.cu
					pfac_file.h
					pfac_file.cpp)

target_compile_features(playground PUBLIC cxx_std_11)

target_compile_features(pfac PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# pfac library to be built with -dc as the member functions 
# could be called by other libraries and executables 
set_property(TARGET pfac 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)