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
text1 = "卫生间抽纸"
text2 = "卫生间抽纸"
text3 = "卫生间抽纸"
text4 = "卫生间抽纸"
batch_text = [text1, text2, text3, text4]
for i in range(len(batch_text), 51):
    batch_text.append(text4)
st = time.time()
inputs = tokenizer(batch_text, return_tensors="pt", padding=True)
logits = model(**inputs).logits
classification = np.argmax(logits.detach().numpy(), axis=-1)
print(f"Python   without loading  cost: {int((time.time()-st)*1000)}ms.")
