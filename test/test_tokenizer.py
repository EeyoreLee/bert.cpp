# -*- encoding: utf-8 -*-
'''
@create_time: 2024/09/03 10:41:08
@author: lichunyu
'''
import warnings

warnings.filterwarnings("ignore")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("/media/E/lichunyu/bert.cpp/data/example_bert_tiny_chinese")
ids = tokenizer("测试一下tokenizer返回的结果是否和python一致")
# print("Python    Ids:", ", ".join(map(str, ids["input_ids"][1:-1])))
print("Python    Ids:", ", ".join(map(str, ids["input_ids"])))
