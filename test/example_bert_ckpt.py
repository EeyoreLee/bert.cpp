# -*- encoding: utf-8 -*-
'''
@create_time: 2024/08/27 09:50:12
@author: lichunyu
'''
import torch
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "/media/E/lichunyu/models/pretrained_models/bert-tiny-chinese",
    num_labels=4)
model.save_pretrained(
    "/media/E/lichunyu/bert.cpp/data/example_bert_tiny_chinese")
...
