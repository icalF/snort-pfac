cmake_minimum_required (VERSION 3.9)
project ("PFAC-CUDA" LANGUAGES CXX CUDA)

add_executable(pfac icalcu.cu kernel.cu)

# particles will also build with -std=c++11
target_compile_features(pfac PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables 
set_property(TARGET pfac 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)