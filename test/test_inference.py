# -*- encoding: utf-8 -*-
'''
@create_time: 2024/09/14 15:10:59
@author: lichunyu
'''
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("/media/E/lichunyu/bert.cpp/data/example_bert_tiny_chinese")
model.eval()
tokenizer = BertTokenizer.from_pretrained("/media/E/lichunyu/bert.cpp/data/example_bert_tiny_chinese")
st = time.time()
inputs = tokenizer("测试一下tokenizer返回的结果是否和python一致", return_tensors="pt")
logits = model(**inputs).logits
classification = np.argmax(logits.detach().numpy())
print(f"Python   classification: {classification} and Python   without loading  cost: {int((time.time()-st)*1000)}ms.")
...
