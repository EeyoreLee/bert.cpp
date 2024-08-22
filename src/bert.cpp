#include "bert.h"
#include "ggml.h"

#include <vector>
#include <fstream>

int bert_predict() {
};

std::vector<int> bert_batch_predict() {
};

bool bert_model_load(const std::string &fname, bert_model &model)
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
        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.num_labels, sizeof(hparams.num_labels));
        fin.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
        printf("%s: num_hidden_layers      = %d\n", __func__, hparams.num_labels);
        printf("%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: ftype                  = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr                  = %d\n", __func__, qntvr);

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
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    case 6:
        wtype = GGML_TYPE_Q5_0;
        break;
    case 7:
        wtype = GGML_TYPE_Q5_1;
        break;
    case 8:
        wtype = GGML_TYPE_Q8_0;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }
    }
};