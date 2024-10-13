#include <string>
#include <chrono>
#include <iostream>

#include "bert.h"

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.ggml";
    const std::string text1 = "卫生间抽纸";
    const std::string text2 = "卫生间抽纸";
    const std::string text3 = "卫生间抽纸";
    const std::string text4 = "卫生间抽纸";

    std::vector<std::string> batch_text{text1, text2, text3, text4};
    for (int i = 4; i <= 50; ++i)
    {
        batch_text.push_back(text4);
    }
    bert_ctx *ctx = new bert_ctx();
    bert_model_load_from_ggml(fname, ctx->model);
    ctx->tokenizer.from_file(path);
    ctx->buf_compute.resize(320 * 1024 * 1024);
    int n_threads = 8;
    auto start_cpp = std::chrono::high_resolution_clock::now();
    std::vector<int> classification = bert_batch_predict(ctx, batch_text, n_threads);
    auto end_cpp = std::chrono::high_resolution_clock::now();
    auto duration_cpp = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp).count();

    std::cout << std::endl
              << std::endl
              << std::endl;

    std::cout << "C++&Rust without loading  cost: " << duration_cpp << "ms." << std::endl;
    std::system("/media/E/lichunyu/miniconda3/envs/bom/bin/python /media/E/lichunyu/bert.cpp/test/test_batch_inference.py");
    return 0;
};