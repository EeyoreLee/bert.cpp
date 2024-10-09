# -*- encoding: utf-8 -*-
'''
@create_time: 2024/09/25 17:59:03
@author: lichunyu
'''
import ctypes
import os
import time


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

    def predict(self):
        ...

    def batch_predict(self, sentences: list):
        n_sentences = len(sentences)
        texts = (ctypes.c_char_p * n_sentences)()
        for i, sentence in enumerate(sentences):
            texts[i] = sentence.encode("utf-8")
        classes = (ctypes.c_int * n_sentences)()
        self.lib.py_bert_batch_predict(self.ctx, texts, n_sentences, self.n_threads, classes)
        classes = [classes[i] for i in range(n_sentences)]
        return classes


if __name__ == "__main__":
    ggml_model_path = "/media/E/lichunyu/bert.cpp/data/ggml-model/ggml_model.gguf"
    tokenizer_json_path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json"
    bertlib = Bert(ggml_model_path=ggml_model_path, tokenizer_json_path=tokenizer_json_path)
    sentences = [
        "测试测试",
        "测试一下返回",
        "测试一下",
        "测试",
        "卫生间抽纸",
    ] * 10
    st = time.time()
    result = bertlib.batch_predict(sentences=sentences)
    print(f"Cost time: {int((time.time()-st)*1000)}ms.")
    print(result)
    ...
