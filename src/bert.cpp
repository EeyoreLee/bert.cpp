#include "bert.h"
#include "ggml.h"

#include <vector>
#include <fstream>

int bert_predict() {
    return 0;
};

std::vector<int> bert_batch_predict() {
    return std::vector<int>{1, 2};
};

bool bert_model_load_from_ggml(const std::string &fname, bert_model &model)
{
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    {
        auto &hparams = model.hparams;

        // TODO adapt the real hparams
        fin.read((char *)&hparams.vocab_size, sizeof(hparams.vocab_size));
        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.intermediate_size, sizeof(hparams.intermediate_size));
        fin.read((char *)&hparams.num_labels, sizeof(hparams.num_labels));
        fin.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fin.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fin.read((char *)&hparams.max_position_embeddings, sizeof(hparams.max_position_embeddings));
        fin.read((char *)&hparams.layer_norm_eps, sizeof(hparams.layer_norm_eps));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: vocab_size              = %d\n", __func__, hparams.vocab_size);
        printf("%s: hidden_size             = %d\n", __func__, hparams.hidden_size);
        printf("%s: intermediate_size       = %d\n", __func__, hparams.intermediate_size);
        printf("%s: num_labels              = %d\n", __func__, hparams.num_labels);
        printf("%s: num_attention_heads     = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: num_hidden_layers       = %d\n", __func__, hparams.num_hidden_layers);
        printf("%s: max_position_embeddings = %d\n", __func__, hparams.max_position_embeddings);
        printf("%s: layer_norm_eps          = %.13f\n", __func__, hparams.layer_norm_eps);
        printf("%s: ftype                   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr                   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.ftype)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }
    }

    const size_t ctx_size = [&]()
    {
        size_t ctx_size = 0;

        const auto &hparams = model.hparams;

        const int32_t hidden_size = hparams.hidden_size;
        const int32_t vocab_size = hparams.vocab_size;
        const int32_t max_position_embeddings = hparams.max_position_embeddings;
        const int32_t num_hidden_layers = hparams.num_hidden_layers;
        const int32_t num_labels = hparams.num_labels;
        const int32_t intermediate_size = hparams.intermediate_size;

        // `BertForSequenceClassification` model
        {
            ctx_size += ggml_row_size(wtype, hidden_size * vocab_size);                            // word_embeddings.weight
            ctx_size += ggml_row_size(wtype, hidden_size * max_position_embeddings);               // position_embeddings.weight
            ctx_size += ggml_row_size(wtype, hidden_size * 2);                                     // token_type_embeddings.weight
            ctx_size += 2 * ggml_row_size(GGML_TYPE_F32, hidden_size);                             // embeddings.LayerNorm.weight + embeddings.LayerNorm.bias
            ctx_size += 3 * num_hidden_layers * ggml_row_size(wtype, hidden_size * hidden_size);   // qkv weight
            ctx_size += 3 * num_hidden_layers * ggml_row_size(GGML_TYPE_F32, hidden_size);         // qkv bias
            ctx_size += num_hidden_layers * ggml_row_size(wtype, hidden_size * hidden_size);       // encoder.layer.*.attention.output.dense.weight
            ctx_size += num_hidden_layers * ggml_row_size(GGML_TYPE_F32, hidden_size);             // encoder.layer.*.attention.output.dense.bias
            ctx_size += num_hidden_layers * 2 * ggml_row_size(GGML_TYPE_F32, hidden_size);         // encoder.layer.*.attention.output.LayerNorm.weight + bias
            ctx_size += num_hidden_layers * ggml_row_size(wtype, intermediate_size * hidden_size); // encoder.layer.*.intermediate.dense.weight
            ctx_size += num_hidden_layers * ggml_row_size(GGML_TYPE_F32, intermediate_size);       // encoder.layer.*.intermediate.dense.bias
            ctx_size += num_hidden_layers * ggml_row_size(wtype, hidden_size * intermediate_size); // encoder.layer.*.output.dense.weight
            ctx_size += num_hidden_layers * ggml_row_size(GGML_TYPE_F32, hidden_size);             // encoder.layer.*.output.dense.bias
            ctx_size += num_hidden_layers * 2 * ggml_row_size(GGML_TYPE_F32, hidden_size);         // encoder.layer.*.output.LayerNorm.weight + bias
            ctx_size += ggml_row_size(GGML_TYPE_F32, hidden_size * hidden_size);                   // pooler.dense.weight
            ctx_size += ggml_row_size(GGML_TYPE_F32, hidden_size);                                 // pooler.dense.bias
            ctx_size += ggml_row_size(wtype, hidden_size * num_labels);                            // classifier.weight
            ctx_size += ggml_row_size(GGML_TYPE_F32, num_labels);                                  // classifier.bias

            ctx_size += (5 + 20 * num_hidden_layers) * 512; // object overhead

            printf("%s: ggml ctx cost %6.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);
        }

        return ctx_size;
    }();

    auto &ctx = model.ctx;

    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        ctx = ggml_init(params);
        if (!ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    {
        const auto &hparams = model.hparams;
        const int32_t num_hidden_layers = hparams.num_hidden_layers;
        const int32_t hidden_size = hparams.hidden_size;
        const int32_t vocab_size = hparams.vocab_size;
        const int32_t max_position_embeddings = hparams.max_position_embeddings;
        const int32_t intermediate_size = hparams.intermediate_size;
        const int32_t num_labels = hparams.num_labels;

        model.encoder.layers.resize(num_hidden_layers);

        // bert_embedding
        {
            model.embedding.word_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, vocab_size);
            model.embedding.position_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, max_position_embeddings);
            model.embedding.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, 2);
            model.embedding.ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            model.embedding.ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // map by name
            model.tensors["bert.embeddings.word_embeddings.weight"] = model.embedding.word_embeddings;
            model.tensors["bert.embeddings.position_embeddings.weight"] = model.embedding.position_embeddings;
            model.tensors["bert.embeddings.token_type_embeddings.weight"] = model.embedding.token_type_embeddings;
            model.tensors["bert.embeddings.LayerNorm.weight"] = model.embedding.ln_w;
            model.tensors["bert.embeddings.LayerNorm.bias"] = model.embedding.ln_b;
        }

        // bert_encoder
        for (int i = 0; i < num_hidden_layers; ++i)
        {
            auto &layer = model.encoder.layers[i];

            // bert_self_attention
            layer.attention.self_attention.q_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.attention.self_attention.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.attention.self_attention.k_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.attention.self_attention.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.attention.self_attention.v_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.attention.self_attention.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // bert_self_output
            layer.attention.self_output.linear_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.attention.self_output.linear_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.attention.self_output.ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.attention.self_output.ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // bert_intermediate
            layer.intermediate.linear_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, intermediate_size);
            layer.intermediate.linear_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, intermediate_size);

            // bert_output
            layer.output.linear_w = ggml_new_tensor_2d(ctx, wtype, intermediate_size, hidden_size);
            layer.output.linear_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.output.ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.output.ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // map by name
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.attention.self_attention.q_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.attention.self_attention.q_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.attention.self_attention.k_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.attention.self_attention.k_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.attention.self_attention.v_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.attention.self_attention.v_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.attention.self_output.linear_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.attention.self_output.linear_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.attention.self_output.ln_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.attention.self_output.ln_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.intermediate.linear_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.intermediate.linear_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.output.linear_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.output.linear_b;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.output.ln_w;
            model.tensors["bert.encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.output.ln_b;
        }

        // bert_pooler
        {
            model.pooler.linear_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            model.pooler.linear_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // map by name
            model.tensors["bert.pooler.dense.weight"] = model.pooler.linear_w;
            model.tensors["bert.pooler.dense.bias"] = model.pooler.linear_b;
        }

        // bert_classifier
        {
            model.classifier.linear_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, num_labels);
            model.classifier.linear_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_labels);

            // map by name
            model.tensors["classifier.weight"] = model.classifier.linear_w;
            model.tensors["classifier.bias"] = model.classifier.linear_b;
        }
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

            if (fin.eof())
            {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0)
            {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));
            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    return true;
};