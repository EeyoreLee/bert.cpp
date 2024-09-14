#include <string>

#include "bert.h"

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.gguf";
    bert_ctx *ctx = new bert_ctx();
    bert_model_load_from_ggml(fname, ctx->model);
    ctx->tokenizer.from_file(path);
    ctx->buf_compute.resize(64 * 1024 * 1024);
    std::string text = "测试一下tokenizer返回的结果是否和python一致";
    int n_threads = 1;
    int r = bert_predict(ctx, text, n_threads);
    return 0;
};