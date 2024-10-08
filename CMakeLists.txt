cmake_minimum_required(VERSION 3.12)
project(bert_cpp)

include(CheckCXXCompilerFlag)

set(CMAKE_BUILD_TYPE "Release")

# test
option(BUILD_TESTS "build test entrance" OFF)


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

if(BUILD_TESTS)
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

  add_executable(test_batch_tokenizer test/test_batch_tokenizer.cpp src/bert.cpp)
  target_include_directories(test_batch_tokenizer PRIVATE ${TOKENZIER_CPP_PATH}/include)
  target_link_libraries(test_batch_tokenizer ggml tokenizers_cpp)

  add_executable(test_batch_inference test/test_batch_inference.cpp src/bert.cpp)
  target_include_directories(test_batch_inference PRIVATE ${TOKENZIER_CPP_PATH}/include)
  target_link_libraries(test_batch_inference ggml tokenizers_cpp)

  add_executable(test_py_entrance test/test_py_entrance.cpp src/bert.cpp)
  target_include_directories(test_py_entrance PRIVATE ${TOKENZIER_CPP_PATH}/include)
  target_link_libraries(test_py_entrance ggml tokenizers_cpp)
endif()

set(BERT_SOURCE src/bert.cpp src/bert.h)
add_library(bert_shared SHARED ${BERT_SOURCE})

target_include_directories(bert_shared PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(bert_shared PUBLIC src)
target_include_directories(bert_shared PRIVATE ggml/include)

target_compile_features(bert_shared PUBLIC cxx_std_20)

target_link_libraries(bert_shared PRIVATE ggml)
target_link_libraries(bert_shared PRIVATE tokenizers_cpp)

set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(bert_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
