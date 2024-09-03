#include "bert.h"

#include <string>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string text = "测试一下tokenizer返回的结果是否和python一致";

    auto start_cpp = std::chrono::high_resolution_clock::now();

    bert_tokenizer *tokenizer = new bert_tokenizer();
    tokenizer->from_file(path);
    std::vector<int> ids = tokenizer->encode(text);
    std::cout << "Test text: " << text << std::endl;
    std::cout << "C++&Rust  Ids: ";
    for (auto i : ids)
    {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    auto end_cpp = std::chrono::high_resolution_clock::now();
    auto duration_cpp = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp).count();

    auto start_python = std::chrono::high_resolution_clock::now();
    std::system("/media/E/lichunyu/miniconda3/envs/bom/bin/python /media/E/lichunyu/bert.cpp/test/test_tokenizer.py");
    auto end_python = std::chrono::high_resolution_clock::now();
    auto duration_python = std::chrono::duration_cast<std::chrono::milliseconds>(end_python - start_python).count();

    std::cout << "C++&Rust  cost: " << duration_cpp << "ms." << std::endl;
    std::cout << "Python    cost: " << duration_python << "ms." << std::endl;

    // free
    delete tokenizer;

    return 0;
};