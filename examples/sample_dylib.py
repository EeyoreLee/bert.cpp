# -*- encoding: utf-8 -*-
'''
@create_time: 2024/09/25 17:59:03
@author: lichunyu
'''
import ctypes
import os
import time

import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer


class Bert:

    def __init__(self,
                 ggml_model_path: str,
                 tokenizer_json_path: str,
                 buf_compute: int = 320,
                 n_threads: int = 8) -> None:
        self.n_threads = n_threads
        self.lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "../build/libbert_shared.so"))

        self.lib.py_bert_ctx_load_from_file.restype = ctypes.c_void_p
        self.lib.py_bert_ctx_load_from_file.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.ctx = self.lib.py_bert_ctx_load_from_file(ggml_model_path.encode("utf-8"),
                                                       tokenizer_json_path.encode("utf-8"), buf_compute)

        self.lib.py_bert_batch_predict.restype = ctypes.POINTER(ctypes.c_int)
        self.lib.py_bert_batch_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        self.num_labels = self.lib.bert_get_num_labels(self.ctx)

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def predict(self):
        ...

    def batch_predict(self, sentences: list[str]):
        n_sentences = len(sentences)
        texts = (ctypes.c_char_p * n_sentences)()
        for i, sentence in enumerate(sentences):
            texts[i] = sentence.encode("utf-8")
        classes = (ctypes.c_int * n_sentences)()
        self.lib.py_bert_batch_predict(self.ctx, texts, n_sentences, self.n_threads, classes)
        classes = [classes[i] for i in range(n_sentences)]
        return classes

    def batch_predict_logits(self, sentences: list[str]):
        n_sentences = len(sentences)
        texts = (ctypes.c_char_p * n_sentences)()
        for i, sentence in enumerate(sentences):
            texts[i] = sentence.encode("utf-8")
        logits = np.zeros((n_sentences, self.num_labels), dtype=np.float32)
        logits_pointers = (ctypes.POINTER(ctypes.c_float) *
                           n_sentences)(*[i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for i in logits])
        self.lib.py_bert_batch_predict_logits(self.ctx, texts, n_sentences, self.n_threads, logits_pointers)
        return logits

    def pytorch_batch_predict_logits(self, sentences: list[str], hf_model_name_or_path: str):
        model = BertForSequenceClassification.from_pretrained(hf_model_name_or_path)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(hf_model_name_or_path)
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        logits = model(**inputs).logits.detach().numpy()
        return logits


if __name__ == "__main__":
    ggml_model_path = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.gguf"
    tokenizer_json_path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json"
    hf_model_name_or_path = "/media/E/lichunyu/bert.cpp/data/example_bert_tiny_chinese"
    bertlib = Bert(ggml_model_path=ggml_model_path, tokenizer_json_path=tokenizer_json_path)
    sentences = [
        "测试测试",
        "测试一下返回",
        "测试一下",
        "测试",
        "卫生间抽纸",
    ] * 10
    st = time.time()
    logits = bertlib.batch_predict_logits(sentences=sentences)
    print(f"Cost time: {int((time.time()-st)*1000)}ms.")
    pytorch_logits = bertlib.pytorch_batch_predict_logits(sentences=sentences,
                                                          hf_model_name_or_path=hf_model_name_or_path)
    print(f"ggml logits:")
    print(logits)
    print()
    print(f"pytorch logits:")
    print(pytorch_logits)
    ...
