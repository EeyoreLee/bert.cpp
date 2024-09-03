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

        bool from_file(const std::string &path);
        std::vector<int> encode(const std::string &text);
    };

    struct bert_ctx
    {
        bert_model model;
        bert_tokenizer tokenizer;
    };

    int bert_predict();
    std::vector<int> bert_batch_predict();
    bool bert_model_load_from_ggml(const std::string &fname, bert_model &model);

#ifdef __cplusplus
}
#endif

#endif // BERT_H