#include "bert.h"

int main(int argc, char **argv)
{
    const char *fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.ggml";
    const char *tokenizer_json_fname = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    int32_t buf_compute = 320;
    bert_ctx *ctx = py_bert_ctx_load_from_file(fname, tokenizer_json_fname, buf_compute);
    int32_t n_threads = 8;
    const char *sentences_arr[] = {"测试一下tokenizer返回的结果是否和python一致", "测试一下tokenizer返回的结果是否", "测试一下tokenizer返回", "测试一下tokenizer返回的结果是否和python"};
    const char **sentences = sentences_arr;
    int32_t n_sentences = 4;
    int classes[n_sentences];
    py_bert_batch_predict(ctx, sentences, n_sentences, n_threads, classes);

    // logits
    int num_labels = bert_get_num_labels(ctx);
    float *logits[n_sentences];
    for (int i = 0; i < n_sentences; i++)
    {
        logits[i] = (float *)malloc(num_labels * sizeof(float));
    }
    py_bert_batch_predict_logits(ctx, sentences, n_sentences, n_threads, logits);
    for (int i = 0; i < n_sentences; i++)
    {
        free(logits[i]);
    }
    return 0;
};