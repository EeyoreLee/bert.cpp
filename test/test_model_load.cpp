#include "bert.h"

#include <string>

int main(int argc, char **argv)
{
    bert_model model;
    std::string fname = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.ggml";
    if (!bert_model_load_from_ggml(fname, model))
    {
        fprintf(stderr, "%s: failed to load model from %s", __func__, fname.c_str());
    }
    return 0;
}