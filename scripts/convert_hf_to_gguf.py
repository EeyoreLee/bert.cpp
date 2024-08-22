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

    state_dict = torch.load(os.path.join(dir_hf_model, "pytorch_model.bin"),
                            map_location="cpu")

    with open(os.path.join(dir_hf_model, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)
    with open(os.path.join(dir_hf_model, "vocab.txt"), "r") as f:
        vocab = f.read().splitlines()
    with open(os.path.join(dir_hf_model, "config.json"), "r") as f:
        config = json.load(f)

    with open(os.path.join(dir_save_model, "ggml_model.gguf"), "wb") as f:

        # hparams
        f.write(struct.pack("i", GGML_FILE_MAGIC))
        f.write(struct.pack("i", config["hidden_size"]))
        # TODO align hparams in bert.h

        for i in range(config["vocab_size"]):
            b_token = bytes(vocab[i], "utf-8")
            f.write(struct.pack("i", len(b_token)))
            f.write(b_token)

        for name, weight in state_dict.items():
            weight = weight.numpy()
            num_dims = len(weight.shape)
            guff_ftype = 1 if ftype == "f32" and num_dims != 1 and name not in [] else 0
            if ftype == "f16":
                weight = weight.astype(np.float16)

            str_name = name.encode("utf-8")
            f.write(struct.pack("iii", num_dims, len(str_name), guff_ftype))
            for dim_size in reversed(weight.shape):
                f.write(struct.pack("i", dim_size))
            weight.tofile(f)

    print(f"Converted on {dir_save_model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_hf_model",
                        type=str,
                        default=None,
                        help="dir for downloaded huggingface pytorch model")
    parser.add_argument("-s", "--save", type=str, default=None)
    parser.add_argument("--ftype", type=str, default="f32", help="f32 or f16")
    args = parser.parse_args()
    convert(args=args)
