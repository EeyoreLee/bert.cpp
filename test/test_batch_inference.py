# -*- encoding: utf-8 -*-
'''
@create_time: 2024/09/23 17:46:25
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
text1 = "测试一下tokenizer返回的结果是否和python一致"
text2 = "测试一下tokenizer返回的结果是否和python"
text3 = "测试一下tokenizer返回的结果是"
text4 = "测试一下tokenizer"
st = time.time()
inputs = tokenizer([text1, text2, text3, text4], return_tensors="pt", padding=True)
logits = model(**inputs).logits
classification = np.argmax(logits.detach().numpy())
print(f"Python   classification: {classification} and Python   without loading  cost: {int((time.time()-st)*1000)}ms.")
