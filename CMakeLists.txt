cmake_minimum_required(VERSION 3.12)
project(test_model_load)

add_subdirectory(ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

include_directories(src)

add_executable(test_model_load test/test_model_load.cpp src/bert.cpp)

target_link_libraries(test_model_load ggml)