# -*- encoding: utf-8 -*-
'''
@create_time: 2024/08/29 19:55:11
@author: lichunyu
'''
import gguf
from transformers import BertTokenizer


class BertModel(object):

    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self) -> None:
        pass

    def write(self):
        ...
