# bert.cpp
BERT inference in C/C++ and Rust

## Get Start
- Make sure `Rust` installed (`curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh` and check it by `rustc -V`)

## Tokenizer performance

|           Tokenizer type            | cost time |
| :---------------------------------: | :-------: |
| transformers.BertTokenizer (Python) |   697ms   |
|    tokenizers-cpp (binding Rust)    |   19ms    |


## Single sentence inference performance
|               Type               | cost time |
| :------------------------------: | :-------: |
|        Python W/O Loading        |   232ms   |
| C++&Rust W/O Loading(n_thread=1) |   48ms    |
| C++&Rust W/O Loading(n_thread=4) |   11ms    |
|         Python W Loading         |  1092ms   |
|  C++&Rust W Loading(n_thread=1)  |   114ms   |
|  C++&Rust W Loading(n_thread=4)  |   79ms    |


## Batch inference performance
|          Type          | cost time |
| :--------------------: | :-------: |
|         Python         |   260ms   |
| C++&Rust (n_thread=8)  |   120ms   |
| C++&Rust (n_thread=32) |   70ms    |

**ggml performance worse as sentence length increases**