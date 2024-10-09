#ifndef BERT_H
#define BERT_H

#include "ggml.h"
#include "tokenizers_cpp.h"

#include <vector>
#include <map>
#include <string>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef int32_t bert_vocab_id;

    struct bert_tokens
    {
        bert_vocab_id *ids;
        size_t size;
    };

    struct bert_batch_tokens
    {
        bert_vocab_id *ids;
        size_t size;
        int32_t batch_size;
        int32_t *attention_mask;

        void init_input_ids(std::vector<std::vector<int>> &input_ids, int32_t pad_id);
    };

    struct bert_hparams
    {
        int32_t vocab_size;
        int32_t hidden_size;
        int32_t intermediate_size;
        int32_t num_labels;
        int32_t num_attention_heads;
        int32_t num_hidden_layers;
        int32_t max_position_embeddings;
        float layer_norm_eps;

        int32_t ftype;
    };

    struct bert_embedding
    {
        struct ggml_tensor *word_embeddings;
        struct ggml_tensor *position_embeddings;
        struct ggml_tensor *token_type_embeddings;
        struct ggml_tensor *ln_w;
        struct ggml_tensor *ln_b;
    };

    struct bert_self_attention
    {
        struct ggml_tensor *q_w;
        struct ggml_tensor *q_b;
        struct ggml_tensor *k_w;
        struct ggml_tensor *k_b;
        struct ggml_tensor *v_w;
        struct ggml_tensor *v_b;
    };

    struct bert_self_output
    {
        struct ggml_tensor *linear_w;
        struct ggml_tensor *linear_b;
        struct ggml_tensor *ln_w;
        struct ggml_tensor *ln_b;
    };

    struct bert_attention
    {
        bert_self_attention self_attention;
        bert_self_output self_output;
    };

    struct bert_intermediate
    {
        struct ggml_tensor *linear_w;
        struct ggml_tensor *linear_b;
    };

    struct bert_output
    {
        struct ggml_tensor *linear_w;
        struct ggml_tensor *linear_b;
        struct ggml_tensor *ln_w;
        struct ggml_tensor *ln_b;
    };

    struct bert_layer
    {
        bert_attention attention;
        bert_intermediate intermediate;
        bert_output output;
    };

    struct bert_encoder
    {
        std::vector<bert_layer> layers;
    };

    struct bert_pooler
    {
        struct ggml_tensor *linear_w;
        struct ggml_tensor *linear_b;
    };

    struct bert_classifier
    {
        struct ggml_tensor *linear_w;
        struct ggml_tensor *linear_b;
    };

    struct bert_model
    {
        bert_hparams hparams;

        bert_embedding embedding;
        bert_encoder encoder;
        bert_pooler pooler;
        bert_classifier classifier;

        struct ggml_context *ctx;
        std::map<std::string, struct ggml_tensor *> tensors;
    };

    struct bert_tokenizer
    {
        std::shared_ptr<tokenizers::Tokenizer> tok;
        int cls_id = 101;
        int seq_id = 102;
        int pad_id = 0;

        bool from_file(const std::string &path);
        std::vector<int> encode(const std::string &text);
        std::vector<std::vector<int>> batch_encode(const std::vector<std::string> &batch_text);
    };

    // Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
    struct bert_buffer
    {
        uint8_t *data = NULL;
        size_t size = 0;

        void resize(size_t size)
        {
            delete[] data;
            data = new uint8_t[size];
            this->size = size;
        }

        ~bert_buffer() { delete[] data; }
    };

    struct bert_ctx
    {
        bert_model model;
        bert_tokenizer tokenizer;
        bert_buffer buf_compute;
    };

    int bert_predict(bert_ctx *ctx, const std::string &text, int32_t n_threads);
    std::vector<int> bert_batch_predict(bert_ctx *ctx, const std::vector<std::string> &text_vec, int32_t n_threads);
    bool bert_model_load_from_ggml(const std::string &fname, bert_model &model);

    // For Python
    bert_ctx *py_bert_ctx_load_from_file(const char *fname, const char *tokenizer_json_fname, int32_t buf_compute);
    void py_bert_batch_predict(bert_ctx *ctx, const char **sentences, int32_t n_sentences, int32_t n_threads, int *classes);

#ifdef __cplusplus
}
#endif

#endif // BERT_H