#include "bert.h"

int main(int argc, char **argv)
{
    const char *fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.gguf";
    const char *tokenizer_json_fname = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    int32_t buf_compute = 320;
    bert_ctx *ctx = py_bert_ctx_load_from_file(fname, tokenizer_json_fname, buf_compute);
    int32_t n_threads = 8;
    const char *sentences_arr[] = {"卫生间抽纸", "卫生间抽纸", "卫生间抽纸", "卫生间抽纸"};
    const char **sentences = sentences_arr;
    int32_t n_sentences = 4;
    int *classification = py_bert_batch_predict(ctx, sentences, n_sentences, n_threads);
    return 0;
};