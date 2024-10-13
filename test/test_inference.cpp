#include <string>
#include <chrono>
#include <iostream>

#include "bert.h"

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.ggml";
    auto start_cpp_total = std::chrono::high_resolution_clock::now();
    bert_ctx *ctx = new bert_ctx();
    bert_model_load_from_ggml(fname, ctx->model);
    ctx->tokenizer.from_file(path);
    ctx->buf_compute.resize(64 * 1024 * 1024);
    auto start_cpp = std::chrono::high_resolution_clock::now();
    std::string text = "测试一下tokenizer返回的结果是否和python一致";
    int n_threads = 4;
    int classification = bert_predict(ctx, text, n_threads);
    auto end_cpp = std::chrono::high_resolution_clock::now();
    auto duration_cpp = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp).count();
    auto duration_cpp_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp_total).count();

    std::cout << std::endl
              << std::endl
              << std::endl;

    std::cout << "C++&Rust classification: " << classification << " and ";
    std::cout << "C++&Rust without loading  cost: " << duration_cpp << "ms." << std::endl;

    auto start_python_total = std::chrono::high_resolution_clock::now();
    std::system("/media/E/lichunyu/miniconda3/envs/bom/bin/python /media/E/lichunyu/bert.cpp/test/test_inference.py");
    auto end_python = std::chrono::high_resolution_clock::now();
    auto duration_python_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_python - start_python_total).count();
    std::cout << "C++&Rust with loading  cost: " << duration_cpp_total << "ms." << std::endl;
    std::cout << "Python   with loading  cost: " << duration_python_total << "ms." << std::endl;
    return 0;
};