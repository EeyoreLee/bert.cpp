cmake_minimum_required(VERSION 3.12)
project(test_model_load)

include(CheckCXXCompilerFlag)

if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

add_subdirectory(ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
set(TOKENZIER_CPP_PATH tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

include_directories(src)
include_directories(ggml/include)
include_directories(tokenizers-cpp/include)

add_executable(test_model_load test/test_model_load.cpp src/bert.cpp)

target_include_directories(test_model_load PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(test_model_load ggml tokenizers_cpp)


add_executable(test_tokenizer test/test_tokenizer.cpp src/bert.cpp)
target_include_directories(test_tokenizer PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(test_tokenizer ggml tokenizers_cpp)


add_executable(test_inference test/test_inference.cpp src/bert.cpp)
target_include_directories(test_inference PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(test_inference ggml tokenizers_cpp)