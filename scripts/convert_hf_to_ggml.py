# -*- encoding: utf-8 -*-
'''
@create_time: 2024/08/22 10:15:37
@author: lichunyu
'''
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGML_FILE_MAGIC = 0x67676d6c


def convert(args):
    dir_save_model = args.save
    Path(dir_save_model).mkdir(parents=True, exist_ok=True)
    dir_hf_model = args.dir_hf_model
    ftype = args.ftype
    if not os.path.exists(dir_hf_model):
        print("Invalid dir_hf_model")
        sys.exit(1)

    state_dict = torch.load(os.path.join(dir_hf_model, "pytorch_model.bin"), map_location="cpu")

    with open(os.path.join(dir_hf_model, "config.json"), "r") as f:
        config = json.load(f)

    with open(os.path.join(dir_save_model, "ggml_model.ggml"), "wb") as f:

        # hparams
        f.write(struct.pack("i", GGML_FILE_MAGIC))
        f.write(struct.pack("i", config["vocab_size"]))
        f.write(struct.pack("i", config["hidden_size"]))
        f.write(struct.pack("i", config["intermediate_size"]))
        f.write(struct.pack("i", len(config["id2label"])))
        f.write(struct.pack("i", config["num_attention_heads"]))
        f.write(struct.pack("i", config["num_hidden_layers"]))
        f.write(struct.pack("i", config["max_position_embeddings"]))
        f.write(struct.pack("f", config["layer_norm_eps"]))
        f.write(struct.pack("i", 0 if ftype == "f32" else 1))
        # TODO align hparams in bert.h

        # for i in range(config["vocab_size"]):
        #     b_token = bytes(vocab[i], "utf-8")
        #     f.write(struct.pack("i", len(b_token)))
        #     f.write(b_token)

        for name, weight in state_dict.items():
            if name in ["bert.embeddings.position_ids"]:
                continue
            weight = weight.numpy()
            num_dims = len(weight.shape)
            gguf_ftype = 1 if ftype == "f16" and num_dims != 1 and name not in [] else 0
            if ftype == "f16":
                weight = weight.astype(np.float16)

            str_name = name.encode("utf-8")
            f.write(struct.pack("iii", num_dims, len(str_name), gguf_ftype))
            for dim_size in reversed(weight.shape):
                f.write(struct.pack("i", dim_size))
            f.write(str_name)
            weight.tofile(f)

    print(f"Converted on {dir_save_model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_hf_model", type=str, default=None, help="dir for downloaded huggingface pytorch model")
    parser.add_argument("-s", "--save", type=str, default=None)
    parser.add_argument("--ftype", type=str, default="f32", help="f32 or f16")
    args = parser.parse_args()
    convert(args=args)
