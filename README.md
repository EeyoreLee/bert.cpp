# bert.cpp
The motivation of this project is to optimize the inference speed of deploying BERT on CPU using PyTorch in Python, while also supporting C++ projects.

## Get Start

Here's a [blog](https://eeyorelee.github.io/model-es-performance-optimization/) but writes in Chinese notes a real case of using the project to optimize inference speed.

### For Python
- Make sure `Rust` installed (`curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh` and check it by `rustc -V`)
- Convert Pytorch checkpoint file and configs to ggml file using 
```
cd scripts/
python convert_hf_to_ggml.py ${dir_hf_model} -s ${dir_saved_ggml_model}
```
- Make sure `tokenizer.json` exists, otherwise execute
```
cd scripts/
python generate_tokenizer_json.py ${dir_hf_model}
```
- Build dynamic library(`libbert_shared.so`)
```
git submodule update --init --recursive
mkdir build
cd build/
cmake ..
make
```
- Refer to `examples/sample_dylib.py`, replace PyTorch inference.


### For C++
- Make sure `Rust` installed (`curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh` and check it by `rustc -V`)
- Convert Pytorch checkpoint file and configs to ggml file using 
```
cd scripts/
python convert_hf_to_ggml.py ${dir_hf_model} -s ${dir_saved_ggml_model}
```
- Make sure `tokenizer.json` exists, otherwise execute
```
cd scripts/
python generate_tokenizer_json.py ${dir_hf_model}
```
- Add this project as a submodule and include it via `add_sub_directory` in your CMake project. You also need to turn on c++17 support. you can then link the library.

## Performance 
### Tokenizer performance

|           Tokenizer type            | cost time |
| :---------------------------------: | :-------: |
| transformers.BertTokenizer (Python) |   734ms   |
|    tokenizers-cpp (binding Rust)    |    3ms    |


### Single sentence inference performance
|               Type               | cost time |
| :------------------------------: | :-------: |
|        Python W/O Loading        |   248ms   |
| C++&Rust W/O Loading(n_thread=4) |    2ms    |
|         Python W Loading         |  1104ms   |
|  C++&Rust W Loading(n_thread=4)  |   19ms    |


### Batch inference performance
|                 Type                 | cost time |
| :----------------------------------: | :-------: |
|        Python(batch_size=50)         |   260ms   |
| C++&Rust (batch_size=50, n_thread=8) |   23ms    |

**ggml performance worse as sentence length increases**

### Python inference using cpp dynamic library
|                    Type                     | cost time |
| :-----------------------------------------: | :-------: |
| Python&C++&Rust (batch_size=50, n_thread=8) |   26ms    |


## Future Work
- Using broadcast instead of `ggml.repeat`. (WIP)
- Update ggml format to gguf.
- Implement Python binding instead of dynamic library.

## Acknowledgements
Thanks for the projects we rely on or refer to.
- [ggerganov/ggml](https://github.com/ggerganov/ggml)
- [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- [monatis/clip.cpp](https://github.com/monatis/clip.cpp)
- [staghado/vit.cpp](https://github.com/staghado/vit.cpp)
- [FFengIll/embedding.cpp](https://github.com/FFengIll/embedding.cpp)
- [skeskinen/bert.cpp](https://github.com/skeskinen/bert.cpp)