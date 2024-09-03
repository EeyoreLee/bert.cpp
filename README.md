# bert.cpp
BERT inference in C/C++ and Rust

## Get Start
- Make sure `Rust` installed (`curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh` and check it by `rustc -V`)

## Tokenizer performance

|           Tokenizer type            | time cost |
| :---------------------------------: | :-------: |
| transformers.BertTokenizer (Python) |   697ms   |
|    tokenizers-cpp (binding Rust)    |   19ms    |