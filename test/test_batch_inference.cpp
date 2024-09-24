#include <string>
#include <chrono>
#include <iostream>

#include "bert.h"

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.gguf";
    const std::string text1 = "测试一下tokenizer返回的结果是否和python一致";
    const std::string text2 = "测试一下tokenizer返回的结果是否和python";
    const std::string text3 = "测试一下tokenizer返回的结果是";
    const std::string text4 = "测试一下tokenizer";

    std::vector<std::string> batch_text{text1, text2, text3, text4};
    bert_ctx *ctx = new bert_ctx();
    bert_model_load_from_ggml(fname, ctx->model);
    ctx->tokenizer.from_file(path);
    ctx->buf_compute.resize(64 * 1024 * 1024);
    int n_threads = 1;
    std::vector<int> classification = bert_batch_predict(ctx, batch_text, n_threads);
    return 0;
};