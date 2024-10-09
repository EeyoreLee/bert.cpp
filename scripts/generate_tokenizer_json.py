# -*- encoding: utf-8 -*-
'''
@create_time: 2024/10/09 19:02:26
@author: lichunyu
'''
import argparse
import os
import sys

from transformers import BertTokenizerFast


def generate(args):
    dir_hf_model = args.dir_hf_model
    if not os.path.exists(dir_hf_model):
        print("Invalid dir_hf_model")
        sys.exit(1)

    tokenizer = BertTokenizerFast.from_pretrained(dir_hf_model)
    tokenizer.save_pretrained(dir_hf_model)

    print(f"Generated on {dir_hf_model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_hf_model", type=str, default=None, help="dir for downloaded huggingface pytorch model")
    args = parser.parse_args()
    generate(args=args)
