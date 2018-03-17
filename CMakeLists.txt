
cmake_minimum_required (VERSION 3.9)
project ("PFAC-CUDA" LANGUAGES CXX CUDA)

#add_executable(ac ac.cpp)

#add_executable(playground icalcu.cu kernel.cu)

add_executable(pfac pfac_main.cu
					pfac.cu
					pfac_match.cu
					pfac_table.cu
					pfac_file.cu
					pfac_kernel.cu)

#target_compile_features(ac PUBLIC cxx_std_11)

#target_compile_features(playground PUBLIC cxx_std_11)

target_compile_features(pfac PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# pfac library to be built with -dc as the member functions 
# could be called by other libraries and executables 
#set_property(TARGET pfac 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)